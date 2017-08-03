from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle
import cv2
from keras import backend as K
from keras.optimizers import Adam #, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils, plot_model 
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

sys.setrecursionlimit(40000)

"""----------------------------------------------------------------------------
                                OPTIONS                              
----------------------------------------------------------------------------"""
# specify parser options when using command line to train.
parser = OptionParser()

parser.add_option("-p", 
                 "--path", 
                  dest="train_path", 
                  help="Path to training data.")

parser.add_option("-o", 
                 "--parser", 
                  dest="parser", 
                  help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")

parser.add_option("-n", 
                  "--num_rois", 
                  dest="num_rois", 
                  help="Number of RoIs to process at once.", 
                  default=32)

parser.add_option("--network", 
                  dest="network", 
                  help="Base network to use. Supports vgg or resnet50.", 
                  default='resnet50')

parser.add_option("--hf", 
                  dest="horizontal_flips", 
                  help="Augment with horizontal flips in training. (Default=false).", 
                  action="store_true", 
                  default=False)

parser.add_option("--vf", 
                  dest="vertical_flips", 
                  help="Augment with vertical flips in training. (Default=false).", 
                  action="store_true", 
                  default=False)

parser.add_option("--rot", "--rot_90", 
                  dest="rot_90", 
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", 
                  default=False)

parser.add_option("--num_epochs", 
                  dest="num_epochs", 
                  help="Number of epochs.", 
                  default=2000)

parser.add_option("--config_filename", 
                  dest="config_filename",
                  help="Location to store all the metadata related to the training (to be used when testing).",
                  default="config.pickle")

parser.add_option("--output_weight_path", 
                  dest="output_weight_path", 
                  help="Output path for weights.", 
                  default='./model_frcnn.hdf5')

parser.add_option("--input_weight_path", 
                  dest="input_weight_path", 
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.")

(options, args) = parser.parse_args()

# options are specified here when testing with spyder to debug. 
# comment these lines when running the script from terminal. 

options.parser =  'simple' 
# options.train_path = 'csv/05_wheelchair_dataset_example.csv'
# options.train_path = 'csv/05_wheelchair_dataset.csv'
options.train_path = 'csv/06_wheelchair_reduced_brix.csv'
options.num_epochs = 100
options.num_rois = 10
# options.input_weight_path = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
options.input_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
# options.input_weight_path = 'vgg_nda_nm_test150.h5'
options.output_weight_path = 'vgg_nda_nm_test.h5'
# options.output_weight_path = 'resnet_nda_nm_test150.h5'
# options.network = 'resnet50'
# options.network = 'vgg'
# verify that arguments were specified correctly. 
if not options.train_path: 
    parser.error('Error: path to training data must be specified. Pass --path to command line')

# parsing data depending on the parser specified.
if options.parser == 'pascal_voc': 
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple': 
    from keras_frcnn.simple_parser import get_data
else: 
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
"""----------------------------------------------------------------------------
                             CONFIGURATION                               
----------------------------------------------------------------------------"""
# pass the settings from the command line, and persist them in the config object
C = config.Config()
C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)
C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)

# import specified network 
if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn
    C.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if options.input_weight_path: 
    C.base_net_weights = options.input_weight_path
else:
    C.base_net_weights = nn.get_weight_path() # set the path to weights based on backend and model

all_imgs, classes_count, class_mapping = get_data(options.train_path)

# check if there is background class specified in the parsed data
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
# inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C,config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'
          .format(config_output_filename))
          
"""----------------------------------------------------------------------------
                            DIVIDING DATASET
----------------------------------------------------------------------------"""
# set it to true if you are starting from a checkpoint 
checkpoint = False

if not checkpoint:
    # divide dataset into train and test
    random.shuffle(all_imgs)
    num_imgs = len(all_imgs)
    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))
    
    # create data generator for train set
    data_gen_train = data_generators.get_anchor_gt(all_imgs, classes_count, C, 
                                                   nn.get_img_output_length, 
                                                   K.image_dim_ordering(), 
                                                   mode='train')
    # create data generator for test set
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, 
                                                nn.get_img_output_length, 
                                                K.image_dim_ordering(), 
                                                mode='test')
    
    # save train and test sets if needed later 
    with open("train.pickle", "wb") as f: 
        pickle.dump(data_gen_train, f)
        print("Saved train generator into {} file".format(f))
    
    with open("val.pickle", "wb") as f: 
        pickle.dump(data_gen_val, f)
        print("Saved train generator into {] file".format(f))
    
else:
    # load sets
    data_gen_trainl = pickle.load(open("train.pickle", "rb"))
    data_gen_val = pickle.load(open("val.pickle", "rb"))
    print("Loaded data generators from pickle files")
    
"""----------------------------------------------------------------------------
                            DEFINING MODELS                                  
----------------------------------------------------------------------------"""
# image settings
input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define optimizers 
optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)

# define RPN & model - returns x_class, x_regrs and base layers 
rpn = nn.rpn(shared_layers, num_anchors)
model_rpn = Model(img_input, rpn[:2])

# define classifier & model - returns out_class and out_regr
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, 
                           nb_classes=len(classes_count), trainable=True)  
model_classifier = Model([img_input, roi_input], classifier)

# define mask & model -  
# mask = nn.mask(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), 
#                trainable=True)         

# model_mask = Model([img_input, roi_input], mask)

# this model holds both RPN and classifier, used to load/save model weights
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

# load model weights
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
    # model_mask.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
        https://github.com/fchollet/keras/tree/master/keras/applications')

# compile models 
model_rpn.compile(optimizer=optimizer, 
                  loss=[losses.rpn_loss_cls(num_anchors), 
                        losses.rpn_loss_regr(num_anchors)])

model_classifier.compile(optimizer=optimizer_classifier, 
                         loss=[losses.class_loss_cls, 
                               losses.class_loss_regr(len(classes_count)-1)],   ## change this 
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

# model_mask.compile(optimizer=optimizer, loss='binary_crossentropy')

model_all.compile(optimizer='sgd', loss='mae')

"""----------------------------------------------------------------------------
                              TRAINING                                   
----------------------------------------------------------------------------"""
# parameters 
epoch_length = 1000    # change to 1000 +++++++++++ DONT FORGET
num_epochs = int(options.num_epochs)
epoch_change = 1.0 /  epoch_length 
vis = True
iter_num = 0
start_time = time.time()
class_mapping_inv = {v: k for k, v in class_mapping.items()}
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
losses = np.zeros((epoch_length, 5))
best_loss = np.Inf
plot_acc_loss = { 'epoch': [],           'rpn_acc': [], 
                  'cls_acc': [],         'loss_rpn_cls': [],   
                  'loss_rpn_regr': [],   'loss_class_cls': [], 
                  'loss_class_regr': [], 'epoch_num': [], 
                  'total_loss':[],       'precision': [], 
                  'recall':[],           'mAP' : []}
                  
print('Starting training')

for epoch_num in range(num_epochs):
    
    cont = raw_input("Continue traninig? [1/0]:")
    if int(cont): pass
    else: break
    
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch: {}/{}'.format(epoch_num + 1, num_epochs))
    
    while True:
        try:
            # end of epoch, calculate av. overlapping bboxes           
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []               
                print('Av. num. of overlapping bboxes from RPN = {} for {} previous iterations'
                      .format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bboxes that overlap the ground truth boxes. Check RPN settings or keep training.')
            
            # get next data generator 
            X, Y, img_data = next(data_gen_train)
          
            # calculate rpn loss             
            loss_rpn = model_rpn.train_on_batch(X, Y)

            # predict on batch             
            P_rpn = model_rpn.predict_on_batch(X)

            # convert rpn to region of interest             
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C,K.image_dim_ordering(), 
                                       use_regr=True, overlap_thresh=0.7, 
                                       max_boxes=300)

            # calculate IoUs - function converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)
            
            if len(neg_samples) > 0: neg_samples = neg_samples[0]
            else: neg_samples = []

            if len(pos_samples) > 0: pos_samples = pos_samples[0]
            else: pos_samples = []
            
            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois//2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                
                try:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)
            
            # calculate losses
            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], 
                                                         [Y1[:, sel_samples, :], 
                                                          Y2[:, sel_samples, :]])
            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]
            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]
            
            # check if there are metrics 
            # print("model metrics rpn \n {} : ".format(model_rpn.metrics))
            # print("model metrics classifier \n {} : ".format(model_classifier.metrics))
            # print("model metrics all \n {} : ".format(model_all.metrics))
            
            # update progrbar with current losses, slows down the process
            progbar.update(iter_num, [('rpn_cls', loss_rpn[1]), ('rpn_regr', loss_rpn[2]),
                                      ('det_cls', loss_class[1]), ('det_regr', loss_class[2])])
            
            # Values to plot
            plot_acc_loss['epoch'].append(epoch_change)
            plot_acc_loss['rpn_acc'].append(len(pos_samples))
            plot_acc_loss['cls_acc'].append(losses[iter_num, 4])
            plot_acc_loss['loss_rpn_cls'].append(losses[iter_num, 0])
            plot_acc_loss['loss_rpn_regr'].append(losses[iter_num, 1])
            plot_acc_loss['loss_class_cls'].append(losses[iter_num, 2])
            plot_acc_loss['loss_class_regr'].append(losses[iter_num, 3])
            epoch_change += epoch_change                 
            
            iter_num += 1
            
            if iter_num == epoch_length:
                
                # calculate epoch mean losses accordingly
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                
                # calculate mean accuracy
                class_acc = np.mean(losses[:, 4])
                
                # Plot rpn losses figure 
                fig_loss = plt.figure(0)
                plt.plot(losses[:, 0], 'r', label='l-rpn cls')  
                plt.plot(losses[:, 1], 'b', label='l-rpn regr')
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
                plt.title('rpn loss during epoch {}'.format(epoch_num + 1), y=1.1)
                plt.ylabel('loss')
                plt.xlabel('epoch length')
                plt.ylim(0, 8)
                plt.grid(True)
                plt.show()
                plt.close()
                fig_loss.savefig('results/loss_rpn_epoch_{}.pdf'.format(epoch_num + 1))
                
                # plot class losses 
                fig_loss = plt.figure(0)
                plt.plot(losses[:, 2], 'g', label='l-class cls')
                plt.plot(losses[:, 3], 'm', label='l-class regr')
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=4, mode="expand", borderaxespad=0.)
                plt.title('cls loss during epoch {}'.format(epoch_num + 1), y=1.1)
                plt.ylabel('loss')
                plt.xlabel('epoch length')
                plt.ylim(0, 8)
                plt.grid(True)
                plt.show()
                plt.close()
                fig_loss.savefig('results/loss_class_epoch_{}.pdf'.format(epoch_num + 1))
                
                # plot accuracy figure 
                fig_acc = plt.figure(0)
                plt.plot(losses[:, 4], 'c', label='accuracy')
                plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=4, mode="expand", borderaxespad=0.)
                plt.title('classifier accuracy for bboxes from RPN  - epoch {}'.format(epoch_num + 1), y=1.1)
                plt.ylabel('accuracy')
                plt.xlabel('epoch length')
                plt.ylim(0, 8)
                plt.grid(True)
                plt.show()
                plt.close()
                fig_acc.savefig('results/acc_epoch_{}.pdf'.format(epoch_num + 1))
            
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()
                
                # Plot average total loss per epoch 
                plot_acc_loss['epoch_num'].append(epoch_num + 1)
                plot_acc_loss['total_loss'].append(curr_loss)
            
                # update changes in loss
                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'
                               .format(best_loss,curr_loss))
                               
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)
                    
                    # epoch chekpoint, not sure it works. 
                    model_all.save('checkpoints/model_epoch_{}.h5'
                                    .format(epoch_num+1)) 
                break
            
        except Exception as e:
            print('Exception: {}'.format(e))
            continue

"""----------------------------------------------------------------------------
                        PLOT GENERAL RESULTS                                  
----------------------------------------------------------------------------"""
# print plot_acc_loss
def plot(num, x, y1, y2, title, ylabel, xlabel, leg1, leg2):
    
    if y2 is not None:        
        # uncomment if using axis limits
        # if max(plot_acc_loss[y1]) > max(plot_acc_loss[y2]):
          #  lim = max(plot_acc_loss[y1])
        # else:
         #   lim = max(plot_acc_loss[y2])
    
        fig = plt.figure(num)
        plt.plot(plot_acc_loss[x], plot_acc_loss[y1], 'r', label=leg1)  
        plt.plot(plot_acc_loss[x], plot_acc_loss[y2], 'b', label=leg2)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
        plt.title(title, y=1.10)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.ylim(0, 8)
        # plt.axis([0, num_epochs + 1, 0, ylim + 1])
        plt.grid(True)
        plt.show()
        plt.close()
    else:
        # uncomment if using axis limits
        # lim = max(plot_acc_loss[y1])
        fig = plt.figure(num)
        plt.plot(plot_acc_loss[x], plot_acc_loss[y1], 'r', label=leg1)  
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
        plt.title(title, y=1.10)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.ylim(0, 8)        
        # plt.axis([0, num_epochs + 1, 0, ylim + 1])
        plt.grid(True)
        plt.show()
        plt.close()
        
    return fig    

# create figures 
fig1 = plot(1, 'epoch', 'rpn_acc', 'cls_acc', 'RPN - classifier accuracy',
            'accuracy', 'epochs', 'overlapping bboxes', 'classifier accuracy')

fig2 = plot(2, 'epoch', 'loss_rpn_cls', 'loss_rpn_regr', 'loss RPN', 'loss', 
               'epochs', 'loss RPN classifier', 'loss RPN regressor') 

fig3 = plot(3, 'epoch', 'loss_class_cls', 'loss_class_regr', 'loss detector', 
               'loss', 'epochs', 'loss detector classifier', 'loss detecor regression') 

fig4 = plot(4, 'epoch_num', 'total_loss', None, 'total loss', 'loss', 'epoch', 
             'total loss', None)

# save figures
fig1.savefig('results/rpn_class_acc.pdf')
fig2.savefig('results/loss_rpn_cls_regr.pdf')
fig3.savefig('results/loss_det_cls_regr.pdf')
fig4.savefig('results/total_loss.pdf')

# plot models
plot_model(model_rpn, to_file='nets/model_rpn.png')
plot_model(model_classifier, to_file='nets/model_classifier.png')
# plot_model(model_mask, to_file='nets/model_mask.png')
plot_model(model_all, to_file='nets/model_all.png')

print('Training complete, exiting.')

"""----------------------------------------------------------------------------
                         CALCULATING MAP FROM TEST SET
----------------------------------------------------------------------------"""
def get_map(pred, gt, f):
    
    T = {}
    P = {}
    fx, fy = f
    
    for bbox in gt: bbox['bbox_matched'] = False
    
    pred_probs = np.array(s['prob'] for s in pred)
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]
    
    # if there are any predictions...
    if pred:
        
        # get predicted and ground truth boxes to calculate IoU
        for box_idx in box_idx_sorted_by_prob:            

            pred_box = pred[box_idx]
            pred_class = pred_box['class']
            pred_x1 = pred_box['x1']
            pred_x2 = pred_box['x2']
            pred_y1 = pred_box['y1']
            pred_y2 = pred_box['y2']
            pred_prob = pred_box['prob']
            
            if pred_class not in P:
                P[pred_class] = []
                T[pred_class] = []

            P[pred_class].append(pred_prob)
            found_match = False
            
            for gt_box in gt:
                
                gt_class = gt_box['class']
                gt_x1 = gt_box['x1']
                gt_x2 = gt_box['x2']
                gt_y1 = gt_box['y1']
                gt_y2 = gt_box['y2']
                gt_seen = gt_box['bbox_matched']
                
                if gt_class != pred_class: continue
                if gt_seen: continue
                
                # calculate intersection ofver union for predicted bounding box
                iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), 
                                          (gt_x1, gt_y1, gt_x2, gt_y2))

                if iou >= 0.5:
                    found_match = True
                    gt_box['bbox_matched'] = True
                    break
                else: continue
                
            T[pred_class].append(int(found_match))
    
    for gt_box in gt:

        if not gt_box['bbox_matched']:
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []                

            P[gt_box['class']].append(1)
            T[gt_box['class']].append(0)
    
    return T, P

# open config.pickle from training 
with open(config_output_filename, 'r') as f_in:
	C = pickle.load(f_in)
    
# turn off any data augmenttaion at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

def format_img(img, C):
    img_min_side = float(C.im_size)
    (h, w, _) = img.shape
    
    if w <= h:
        f = img_min_side / w
        nh = int(f * h) # new height
        nw = int(img_min_side) # new width 
    else:
        f = img_min_side / h
        nw = int(f * w)
        nh = int(img_min_side)
    
    fx = w / float(nw)
    fy = h / float(nh)
    
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]    
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img, fx, fy

# class mapping and image shaping
class_mapping = {v: k for k, v in class_mapping.iteritems()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}    
C.num_rois = int(options.num_rois)

if C.network == 'resnet50': num_features = 1024
elif C.network == 'vgg': num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

# img_input = Input(shape=input_shape_img)
# roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the test models
rpn_l = nn.rpn(shared_layers, num_anchors)
class_l = nn.classifier(feature_map_input, roi_input, C.num_rois, 
                        nb_classes=len(class_mapping), trainable=True)
model_rpn_l = Model(img_input, rpn_l)
model_classifier_lo = Model([feature_map_input, roi_input], class_l)
model_classifier_l = Model([feature_map_input, roi_input], class_l)

model_rpn_l.load_weights(C.model_path, by_name=True)
model_classifier_l.load_weights(C.model_path, by_name=True)

model_rpn_l.compile(optimizer='sgd', loss='mse')
model_classifier_l.compile(optimizer='sgd', loss='mse')

# do the magic 
bbox_threshold = 0.0
visualize = True
T = {}
P = {}

for idx in range(len(val_imgs)):
    
    # get image data
    print('{}/{}'.format(idx, len(val_imgs)))
    st = time.time()    
    X, Y, img_data = next(data_gen_val)
    filepath = img_data['filepath']
    img = cv2.imread(filepath)
    X, fx, fy = format_img(img, C)
    
    # scale image
    img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
    img_scaled[:, :, 0] += 123.680
    img_scaled[:, :, 1] += 116.779
    img_scaled[:, :, 2] += 103.939
    
    img_scaled = img_scaled.astype(np.uint8)
    
    if K.image_dim_ordering() == 'tf': X = np.transpose(X, (0, 2, 3, 1))
        
    # get the feature maps and output from rpn 
    [Y1, Y2, F] = model_rpn_l.predict(X)
    
    # convert rpn to roi
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)
    
    # convert from (x1, y1, x2, y2) to (x, y, w, h)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]
    
    # apply the spp to the proposed regions 
    bboxes = {}
    probs = {}
    
    for jk in range(R.shape[0] // C.num_rois + 1):
        ROIs = np.expand_dims(R[C.num_rois * jk: C.num_rois * (jk +1), :], axis=0)
        
        if ROIs.shape[1] == 0: break
        
        if jk == R.shape[0] // C.num_rois:
            
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded
        
        [P_cls, P_regr] = model_classifier_lo.predict([F, ROIs])
        
        for ii in range(P_cls.shape[1]):
            
            if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue
            
            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
            
            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []
                
            (x, y, w, h) = ROIs[0, ii, :]            
            cls_num = np.argmax(P_cls[0, ii, :])
            
            try:
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except: pass
            
            bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))
            
        all_dets = []
        
        for key in bboxes:            
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            
            for jk in range(new_boxes.shape[0]):                
                (x1, y1, x2, y2) = new_boxes[jk, :]                
                # bounding box 
                cv2.rectangle(img_scaled, 
                              (x1, y1), 
                              (x2, y2), 
                              class_to_color[key], 2)                    
                # text     
                textLabel = '{}: {}'.format(key, float("{0:.4f}".format(new_probs[jk])))
                det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
                all_dets.append(det)                
                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                textOrg = (x1, y1 + 20)                
                # rectangle label
                cv2.rectangle(img_scaled, 
                             (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                             (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), 
                             (0, 0, 0), 2)                
                # rectangle label
                cv2.rectangle(img_scaled, 
                             (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                             (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), 
                             class_to_color[key], -1)                              
                # add label class
                cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
        print('Elapsed time = {}'.format(time.time() - st))        
        t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))        

        for key in t.keys():
            if key not in T:
                T[key] = []
                P[key] = []
            T[key].extend(t[key])
            P[key].extend(p[key])
        
        all_aps = []
        
        for key in T.keys():
            ap = average_precision_score(T[key], P[key])
            print('{} AP: {}'.format(key, ap))
            all_aps.append(ap)
        
        print('mAP = {}'.format(np.mean(np.array(all_aps))))        
        plot_acc_loss['mAP'].append(np.mean(np.array(all_aps)))
        
    print(all_dets)
    cv2.imshow('img', img_scaled)
    cv2.waitKey(0)
    cv2.imwrite('./results/detected_mAP{}.png'.format(idx), img_scaled)

# plot mAP
lim = (max(plot_acc_loss['mAP']))
fig_mAP = plt.figure(0)
plt.plot(plot_acc_loss['mAP'], 'r', label='mAP')  
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                           ncol=2, mode="expand", borderaxespad=0.)
plt.title('mAP'.format(epoch_num + 1), y=1.1)
plt.ylabel('mAP')
plt.xlabel('iter')
plt.ylim(0, 2)
plt.grid(True)
plt.show()
plt.close()
fig_loss.savefig('results/mAP.pdf'.format(epoch_num + 1))