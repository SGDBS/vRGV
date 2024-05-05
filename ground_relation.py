# # ====================================================
# # @Author  : Xiao Junbin
# # @Email   : junbin@comp.nus.edu.sg
# # @File    : ground_relation.py
# # ====================================================
# # from networks.basic import *
# import numpy as np
# from networks.relation2relation import *
# from utils import *
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import os.path as osp
# import time
# import json
# import pickle
# import nltk
# import csv
# from torch.nn.utils import clip_grad_value_
# from torch.nn.functional import kl_div
#
#
#
# bd_motions = {'beneath': 'above', 'above': 'beneath', 'right': 'left', 'left': 'right', 'behind': 'front', 'front': 'behind', 'next_to': 'next_to', 'outside': 'inside', 'inside': 'outside', 'shorter': 'taller', 'taller': 'shorter', 'smaller': 'larger', 'larger': 'smaller', 'lie_beneath': 'lie_above', 'lie_above': 'lie_beneath', 'lie_right': 'lie_left', 'lie_left': 'lie_right', 'lie_outside': 'lie_inside', 'lie_inside': 'lie_outside', 'lie_next_to': 'lie_next_to', 'stand_beneath': 'stand_above', 'stand_above': 'stand_beneath', 'stand_right': 'stand_left', 'stand_left': 'stand_right', 'stand_behind': 'stand_front', 'stand_front': 'stand_behind', 'stand_next_to': 'stand_next_to', 'stand_outside': 'stand_inside', 'stand_inside': 'stand_outside', 'sit_beneath': 'sit_above', 'sit_above': 'sit_beneath', 'sit_right': 'sit_left', 'sit_left': 'sit_right', 'sit_behind': 'sit_front', 'sit_front': 'sit_behind', 'sit_next_to': 'sit_next_to', 'sit_outside': 'sit_inside', 'sit_inside': 'sit_outside'}
#
#
#
# class GroundRelation():
#     def __init__(self, vocab, train_loader, val_loader, checkpoint_path, model_prefix, vis_step, save_step, visual_dim, lr, batch_size, epoch_num, cuda):
#
#         self.vocab = vocab
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.model_dir = checkpoint_path
#         self.model_name = model_prefix
#         self.vis_step = vis_step
#         self.save_step = save_step
#
#         self.lr = lr
#         self.grad_clip = 10
#         self.batch_size = batch_size
#         self.epoch_num = epoch_num
#         self.cuda = cuda
#
#         self.input_size = 512
#         self.hidden_size = 512
#         self.visual_dim = visual_dim
#         self.word_dim = 300
#
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
#
#         with open("./dataset/vidvrd/vocab.pkl", 'rb') as fp:
#             self.vocab = pkl.load(fp)
#
#     ################################################
#     # add vocab to build reverse relation tokens
#     def get_word_idx(self, relation):
#         """
#         convert relation to index sequence
#         :param relation:
#         :return:
#         """
#         # relation = relation.split('-')
#         # relation = '-'.join([relation[0],relation[-1]])
#
#         table = str.maketrans('-_', '  ')
#         # print("relation:",relation)
#         relation_trans = relation.translate(table)
#
#         tokens = nltk.tokenize.word_tokenize(str(relation_trans).lower())
#         relation_token = []
#         relation_token.append(self.vocab('<start>'))
#         relation_token.extend([self.vocab(token) for token in tokens])
#         relation_token.append(self.vocab('<end>'))
#         target = torch.Tensor(relation_token)
#
#         return target
#     ################################################
#
#
#     def build_model(self):
#         self.relation_ground = AttHierarchicalGround(self.input_size, self.hidden_size, self.visual_dim, self.word_dim)
#
#         self.relation_reconstruction = DecoderRNN(self.input_size, self.hidden_size, len(self.vocab), 1, 10)
#
#         params = [{'params':self.relation_reconstruction.parameters()}, {'params':self.relation_ground.parameters()}]
#         self.optimizer = torch.optim.Adam(params=params,
#                                              lr=self.lr)
#
#
#         self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=2, verbose=True)
#
#         self.relation_ground.to(self.device)
#         self.relation_reconstruction.to(self.device)
#         self.criterion = nn.CrossEntropyLoss().to(self.device)
#
#     def save_model(self, epoch):
#         torch.save(self.relation_reconstruction.state_dict(),
#                    osp.join(self.model_dir, '{}-transformer-reconstruct-{}.ckpt'.format(self.model_name,"retrain-temporal-stage2-epoch{}").format(epoch)))
#         print("reconstruct model saved at {}".format(osp.join(self.model_dir, '{}-transformer-reconstruct-{}.ckpt'.format(self.model_name, "retrain-temporal-stage2-epoch{}").format(epoch))))
#         torch.save(self.relation_ground.state_dict(),
#                    osp.join(self.model_dir, '{}-transformer-ground-{}.ckpt'.format(self.model_name, "retrain-temporal-stage2-epoch{}").format(epoch)))
#         print("ground model saved at {}".format(osp.join(self.model_dir, '{}-transformer-ground-{}.ckpt'.format(self.model_name, "retrain-temporal-stage2-epoch{}").format(epoch))))
#
#     def resume(self):
#         """
#         Initialize with the basic model obtained at the 1st stage.
#         :param epoch:
#         :return:
#         """
#         # ground_model_file = osp.join(self.model_dir,'vRGV-basic-ground-3.ckpt')
#         # reconstruct_model_file = osp.join(self.model_dir,'vRGV-basic-reconstruct-3.ckpt')
#
#         # ground_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-ground-best-stage1-epoch1.ckpt"
#         # reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-best-stage1-epoch1.ckpt"
#
#         # ground_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-ground-retrain-stage2-epoch3.ckpt"
#         # reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-retrain-stage2-epoch3.ckpt"
#
#         reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-retrain-stage1-epoch1.ckpt"
#
#         # ground_dict = torch.load(ground_model_file)
#         # print("ground model loaded, at {}".format(ground_model_file))
#         reconstruct_dict = torch.load(reconstruct_model_file)
#         print("reconstruct model loaded, at {}".format(reconstruct_model_file))
#
#         # new_ground_dict = {}
#         # for k, v in self.relation_ground.state_dict().items():
#         #    if k in ground_dict:
#         #        v = ground_dict[k]
#         #    new_ground_dict[k] = v
#         # self.relation_ground.load_state_dict(new_ground_dict) #only reconstruction part is better
#
#         new_reconstruct_dict = {}
#         for k, v in self.relation_reconstruction.state_dict().items():
#             if k in reconstruct_dict:
#                 v = reconstruct_dict[k]
#             new_reconstruct_dict[k] = v
#
#         self.relation_reconstruction.load_state_dict(new_reconstruct_dict)
#
#
#     def run(self, pretrain=False):
#
#         self.build_model()
#         if pretrain:
#             self.resume()
#             print("pretrained model loaded(form ground_relation.py)")
#         else:
#             print("start training from scratch(from ground_relation.py)")
#         #self.relation_reconstruction.position = self.relation_ground.position
#         ################################
#         #print("loading pretrained weights but the original weights do NOT include the Transformer Part")
#         #self.relation_ground.load_state_dict(torch.load("/home/gpu4/Vigelos/vRGV_data/vRGV/models/vidvrd/visual_bbox_trans_temp2-ground-6.ckpt"),strict=False)
#         #self.relation_reconstruction.load_state_dict(torch.load("/home/gpu4/Vigelos/vRGV_data/vRGV/models/vidvrd/visual_bbox_trans_temp2-reconstruct-6.ckpt"),strict=False)
#         ################################
#
#         save_loss = np.inf
#
#         for epoch in range(0, self.epoch_num):
#             train_loss = self.train(epoch)
#             val_loss = self.val(epoch)
#
#             print('==> Epoch:[{}/{}] [Training loss: {:.4f} {:.4f} Val loss: {:.4f} {:.4f}]'.
#                   format(epoch, self.epoch_num, train_loss, np.exp(train_loss), val_loss, np.exp(val_loss)))
#
#             self.scheduler.step(val_loss)
#
#             if val_loss < save_loss:
#                 save_loss = val_loss
#                 self.save_model(epoch)
#
#
#     def inspect(self):
#          self.build_model()
#          ground_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-ground-6.ckpt"
#          reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-reconstruct-6.ckpt"
#
#          # ground_dict = torch.load(ground_model_file)
#          # print("ground model loaded, at {}".format(ground_model_file))
#          reconstruct_dict = torch.load(reconstruct_model_file)
#          print("reconstruct model loaded, at {}".format(reconstruct_model_file))
#
#          # new_ground_dict = {}
#          # for k, v in self.relation_ground.state_dict().items():
#          #    if k in ground_dict:
#          #        v = ground_dict[k]
#          #    new_ground_dict[k] = v
#          # self.relation_ground.load_state_dict(new_ground_dict) #only reconstruction part is better
#
#          new_reconstruct_dict = {}
#          for k, v in self.relation_reconstruction.state_dict().items():
#              if k in reconstruct_dict:
#                 v = reconstruct_dict[k]
#                 new_reconstruct_dict[k] = v
#          self.relation_reconstruction.load_state_dict(new_reconstruct_dict)
#          print(self.val(0))
#
#
#
#
#     def train(self, epoch):
#         print('==> Epoch:[{}/{}] [training stage Encode_lr: {} Decode_lr: {}]'.
#               format(epoch, self.epoch_num, self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[0]['lr']))
#
#         self.relation_ground.train()
#         self.relation_reconstruction.train()
#
#         total_step = len(self.train_loader)
#         epoch_loss = 0
#
#         # bd_sub_attn = []
#         # bd_obj_attn = []
#         # bd_beta1 = []
#         # bd_beta2 = []
#         # bd_video_name = []
#         # bd_relation_text = []
#
#         for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.train_loader):
#
#             videos = videos.to(self.device)
#             relations = relations.to(self.device)
#
#             # print("text:",len(relation_text))
#
#             # ################################
#             # # prepare the reverse relation
#             # reverse_relation_text = []
#             # video_num = []
#             #
#             # # filter all the bio-directional motions
#             # for i in range(len(relation_text)):
#             #     motion = relation_text[i].split("-")[1]
#             #     if motion in bd_motions:
#             #         video_num.append(i)
#             #         reverse_relation = "-".join([relation_text[i].split("-")[2], bd_motions[relation_text[i].split("-")[1]],  relation_text[i].split("-")[0]])
#             #         reverse_relation_text.append(reverse_relation)
#             # bdvideo = torch.zeros([len(video_num),120,40,2053]).to(videos.device)
#             # reverse_valid_lengths = []
#             # for i in range(len(video_num)):
#             #     bdvideo[i] = videos[video_num[i]]
#             #     reverse_valid_lengths.append(valid_lengths[video_num[i]])
#             #
#             # reverse_video_out,reverse_video_hidden, reverse_sub_attn, reverse_obj_attn,reverse_beta1, reverse_beta2 = self.relation_ground(bdvideo,reverse_relation_text)
#             # ###################################
#
#             targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]
#
#             video_out, video_hidden, sub_attn, obj_attn, normal_beta1,  normal_beta2 = self.relation_ground(videos, relation_text)
#             # video_out, video_hidden = self.relation_ground(videos, relation_text)
#
#
#             relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)
#
#             ##########################################
#             # filter all the sample of bd motions
#             # bd_vid = []
#             # for id in range(len(relation_text)):
#             #     if relation_text[id].split("-")[1] in bd_motions:
#             #         bd_vid.append(id)
#             # bd_sub_attn.extend([sub_attn[vid] for vid in bd_vid])
#             # bd_obj_attn.extend([obj_attn[vid] for vid in bd_vid])
#             # bd_beta1.extend([normal_beta1[vid] for vid in bd_vid])
#             # bd_beta2.extend([normal_beta2[vid] for vid in bd_vid])
#             # bd_video_name.extend([video_names[vid] for vid in bd_vid])
#             # bd_relation_text.extend([relation_text[vid] for vid in bd_vid])
#             ##########################################
#
#             # rct_loss_normal = self.criterion(relation_decode, targets)
#             rct_loss = self.criterion(relation_decode, targets)
#
#
#             # #########################################################
#             # # add reverse relation to reconstruction loss
#             # reverse_relation = [self.get_word_idx(subpredobj) for subpredobj in reverse_relation_text]
#             # # merge relations
#             # reverse_lengths = [len(rel) for rel in reverse_relation]
#             # reverse_relation_token = torch.zeros(len(reverse_relation), max(reverse_lengths)).long().to(relations.device)
#             # for i, rel in enumerate(reverse_relation):
#             #     end = reverse_lengths[i]
#             #     reverse_relation_token[i, :end] = rel[:end]
#             # reverse_relation_decode = self.relation_reconstruction(reverse_video_out, reverse_video_hidden, reverse_relation_token, reverse_valid_lengths)
#             # reverse_targets = pack_padded_sequence(reverse_relation_token, reverse_valid_lengths, batch_first=True)[0]
#             # rct_loss_reverse = self.criterion(reverse_relation_decode, reverse_targets)
#             # rct_loss = (rct_loss_normal + rct_loss_reverse)
#             #########################################################
#
#
#             ############################################
#             # compute kl loss between normal and reverse video feature
#             # normal_beta1 = torch.stack([normal_beta1[i] for i in video_num],dim=0)
#             # normal_beta2 = torch.stack([normal_beta2[i] for i in video_num],dim=0)
#             # normal_beta = torch.zeros_like(normal_beta1)
#             # reverse_beta = torch.zeros_like(reverse_beta1)
#             #
#             # for i in range(120):
#             #     normal_beta[:,i] = normal_beta1[:,i] + normal_beta2[:,i//12]
#             #     reverse_beta[:,i] = reverse_beta1[:,i] + reverse_beta2[:,i//12]
#             #
#             # kl_loss = kl_div(normal_beta.log(), reverse_beta, reduction='sum')
#             ################################################
#
#
#             self.relation_ground.zero_grad()
#             self.relation_reconstruction.zero_grad()
#
#             # loss = (rct_loss+kl_loss+clr_loss)
#             # loss = (rct_loss+kl_loss)
#             # loss = kl_loss
#             loss = rct_loss
#             loss.backward()
#
#             # clip_gradient(self.optimizer, self.grad_clip)
#             # clip_grad_value_()
#             self.optimizer.step()
#
#             cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#
#
#             if iter % self.vis_step == 0:
#
#                 print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
#                       format(iter, total_step, cur_time, rct_loss.item(), np.exp(loss.item())))
#                 # print("    rct_loss:", rct_loss.item(), "kl_loss:", kl_loss.item(), "clr_loss:", clr_loss.item())
#                 # print("    rct_loss:", rct_loss.item(), "kl_loss:", kl_loss.item())
#                 # print("    kl_loss:", kl_loss.item())
#                 print("    rct_loss:", rct_loss.item())
#
#
#         # data = [(bd_video_name[i],bd_relation_text[i],bd_sub_attn[i],bd_obj_attn[i],bd_beta1[i],bd_beta2[i]) for i in range(len(bd_video_name))]
#         # with open('data-epoch-{}.pickle'.format(epoch), 'wb') as f:
#         #     pickle.dump(data, f)
#
#         return epoch_loss / total_step
#
#
#     def val(self, epoch):
#         print('==> Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))
#
#         self.relation_ground.eval()
#         self.relation_reconstruction.eval()
#
#         total_step = len(self.val_loader)
#         epoch_loss = 0
#
#         correct_count=0
#         all_count=0
#
#         with torch.no_grad():
#             for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):
#                 # print("relation text:",relation_text)
#                 # print("videos:",videos.shape)
#                 # print("relations token:",relations)
#                 # print("valid length:",valid_lengths)
#                 # print("video name:",video_names)
#                 videos = videos.to(self.device)
#                 relations = relations.to(self.device)
#                 targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]
#
#                 video_out, video_hidden,_,_a,_b,_c = self.relation_ground(videos, relation_text)
#                 # video_out, video_hidden = self.relation_ground(videos, relation_text)
#
#                 relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)
#
#                 loss = self.criterion(relation_decode,targets)
#
#                 cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                 if iter % self.vis_step == 0:
#                     print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
#                           format(iter, total_step, cur_time,  loss.item(), np.exp(loss.item())))
#
#                 epoch_loss += loss.item()
#
#         return epoch_loss / total_step
#
#
#     def predict(self, ep):
#
#         self.build_model()
#
#         ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
#         reconstruction_path = osp.join(self.model_dir, '{}-reconstruct-{}.ckpt'.format(self.model_name, ep))
#
#         self.relation_reconstruction.eval()
#         self.relation_ground.eval()
#
#         self.relation_ground.load_state_dict(torch.load(ground_model_path))
#         self.relation_reconstruction.load_state_dict(torch.load(reconstruction_path))
#         total = len(self.val_loader)
#         pos_num = 0
#
#         fout = open('results/prediction.txt', 'w')
#         with torch.no_grad():
#             for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):
#
#                 videos = videos.to(self.device)
#
#                 video_out, video_hidden = self.relation_ground(videos, relation_text)
#
#                 sample_ids = self.relation_reconstruction.sample(video_out, video_hidden)
#
#                 sample_ids = sample_ids[0].cpu().numpy()
#
#                 predict_relation = []
#                 for id in sample_ids:
#                     word = self.vocab.idx2word[id]
#                     predict_relation.append(word)
#                     if word == '<end>': break
#
#                 predict_relation = ' '.join(predict_relation[1:-1])
#                 # print(relation_text[0], predict_relation)
#
#                 table = str.maketrans('-_', '  ')
#                 relation = relation_text[0].translate(table)
#                 output_str = "{}:{}".format(relation, predict_relation)
#                 fout.writelines(output_str+'\n')
#
#                 if relation == predict_relation:
#                     pos_num += 1
#
#                 if iter%self.vis_step == 0:
#                     print("{}:{}".format(iter, output_str))
#
#         print("Reconstrution Rate: ", pos_num / total)
#         fout.close()
#
#
#     def ground_attention(self, ep, save_name):
#         """output the spatial temporal attention as grounding results"""
#         self.build_model()
#         # ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
#         self.relation_ground.eval()
#         ground_model_path = "models/vidvrd/acc26.88.ckpt"
#         # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-retrain-aug-stage2-epoch5.ckpt"
#         # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-retrain-onlyrct-aug-stage2-epoch0.ckpt"
#         # ground_model_path = "models/vidvrd/acc26.12.beta0008.transformer.base.ckpt"
#
#         # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-ground-6.ckpt"
#         # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-retrain-temporal-stage2-epoch6.ckpt"
#
#         self.relation_ground.load_state_dict(torch.load(ground_model_path))
#         print("load ground model from {}(from ground_relation.py)".format(ground_model_path))
#         video_res = {}
#
#         total = len(self.val_loader)
#
#         with open("dataset/vidvrd/gt_relation_frame.json") as f:
#             gt = json.load(f)
#
#         with torch.no_grad():
#             for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):
#
#                 videos = videos.to(self.device)
#                 sub_atts, obj_atts, beta1, beta2 = self.relation_ground(videos, relation_text, mode='val')
#                 # print(sub_atts.shape, beta1.shape) #(32, 120, 40), (32, 120)
#
#                 data_sub_atts = sub_atts.data.cpu().numpy()
#                 data_obj_atts = obj_atts.data.cpu().numpy()
#                 data_beta2 = beta2.data.cpu().numpy()
#                 data_beta1 = beta1.data.cpu().numpy()
#                 real_bs = data_sub_atts.shape[0]
#
#
#                 for bs in range(real_bs):
#                     data = {}
#                     data['sub'] = data_sub_atts[bs].tolist()
#                     data['obj'] = data_obj_atts[bs].tolist()
#                     data['beta2'] = data_beta2[bs].tolist()
#                     data['beta1'] = data_beta1[bs].tolist()
#
#                     vname = video_names[bs]
#                     if vname not in video_res:
#                         video_res[vname] = {}
#                     video_res[vname][relation_text[bs]] = data
#
#
#
#
#                 if iter % self.vis_step == 0:
#                     print('Finished: {}-{}'.format(iter, total))
#
#
#             save_results(save_name, video_res)
#             # with open('correct_sample.json', 'r') as f:
#             #     correct_sample = json.load(f)
#             # with open("dataset/vidvrd/gt_relation_frame.json") as f:
#             #     gt_result = json.load(f)
#
#             # count = 0
#             # for vname in gt_result.keys():
#             #     for relation in gt_result[vname].keys():
#             #         if ([vname,relation] not in correct_sample):
#             #             b1 = video_res[vname][relation]["beta1"]
#             #             b2 = video_res[vname][relation]["beta2"]
#             #             start = list(gt_result[vname][relation][0]['obj'].keys())[0]
#             #             end = list(gt_result[vname][relation][0]['obj'].keys())[-1]
#             #             b = torch.zeros([120])
#             #
#             #             for i in range(120):
#             #                 b[i] = b1[i] + b2[i // 12]
#             #             b = b.tolist()
#             #
#             #             with open('{}-{}-{}-{}.csv'.format(vname, relation, start, end),
#             #                       'w', newline='') as f:
#             #                 writer = csv.writer(f)
#             #                 writer.writerow(b)
#             #             count+=1
#             #             if count>20:
#             #                 break
#             #         else:
#             #             break
#
#
#             # for g in range(10):
#             #     num = g*20
#             #     b1 = video_res[correct_sample[num][0]][correct_sample[num][1]]["beta1"]
#             #     b2 = video_res[correct_sample[num][0]][correct_sample[num][1]]["beta2"]
#             #     start = list(gt_result[correct_sample[num][0]][correct_sample[num][1]][0]['obj'].keys())[0]
#             #     end = list(gt_result[correct_sample[num][0]][correct_sample[num][1]][0]['obj'].keys())[-1]
#             #     b = torch.zeros([120])
#             #
#             #     for i in range(120):
#             #         b[i] = b1[i] + b2[i // 12]
#             #     b = b.tolist()
#             #
#             #     print(num)
#             #     with open('{}-{}-{}-{}.csv'.format(correct_sample[num][0],correct_sample[num][1],start,end), 'w', newline='') as f:
#             #         writer = csv.writer(f)
#             #         writer.writerow(b)
#
#
#
#
#


################################################################
# YYS version
# ====================================================
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : ground_relation.py
# ====================================================
# from networks.basic import *
import numpy as np
from networks.relation2relation import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os.path as osp
import time
import pickle
import nltk
from torch.nn.utils import clip_grad_value_
from torch.nn.functional import kl_div

bd_motions = {'beneath': 'above', 'above': 'beneath', 'right': 'left', 'left': 'right', 'behind': 'front',
              'front': 'behind', 'next_to': 'next_to', 'outside': 'inside', 'inside': 'outside', 'shorter': 'taller',
              'taller': 'shorter', 'smaller': 'larger', 'larger': 'smaller', 'lie_beneath': 'lie_above',
              'lie_above': 'lie_beneath', 'lie_right': 'lie_left', 'lie_left': 'lie_right', 'lie_outside': 'lie_inside',
              'lie_inside': 'lie_outside', 'lie_next_to': 'lie_next_to', 'stand_beneath': 'stand_above',
              'stand_above': 'stand_beneath', 'stand_right': 'stand_left', 'stand_left': 'stand_right',
              'stand_behind': 'stand_front', 'stand_front': 'stand_behind', 'stand_next_to': 'stand_next_to',
              'stand_outside': 'stand_inside', 'stand_inside': 'stand_outside', 'sit_beneath': 'sit_above',
              'sit_above': 'sit_beneath', 'sit_right': 'sit_left', 'sit_left': 'sit_right', 'sit_behind': 'sit_front',
              'sit_front': 'sit_behind', 'sit_next_to': 'sit_next_to', 'sit_outside': 'sit_inside',
              'sit_inside': 'sit_outside'}


class GroundRelation():
    def __init__(self, vocab, train_loader, val_loader, checkpoint_path, model_prefix, vis_step, save_step, visual_dim,
                 lr, batch_size, epoch_num, cuda):

        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_dir = checkpoint_path
        self.model_name = model_prefix
        self.vis_step = vis_step
        self.save_step = save_step

        self.lr = lr
        self.grad_clip = 10
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.cuda = cuda

        self.input_size = 512
        self.hidden_size = 512
        self.visual_dim = visual_dim
        self.word_dim = 300

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

        with open("./dataset/vidvrd/vocab.pkl", 'rb') as fp:
            self.vocab = pkl.load(fp)

    ################################################
    # add vocab to build reverse relation tokens
    def get_word_idx(self, relation):
        """
        convert relation to index sequence
        :param relation:
        :return:
        """
        # relation = relation.split('-')
        # relation = '-'.join([relation[0],relation[-1]])

        table = str.maketrans('-_', '  ')
        # print("relation:",relation)
        relation_trans = relation.translate(table)

        tokens = nltk.tokenize.word_tokenize(str(relation_trans).lower())
        relation_token = []
        relation_token.append(self.vocab('<start>'))
        relation_token.extend([self.vocab(token) for token in tokens])
        relation_token.append(self.vocab('<end>'))
        target = torch.Tensor(relation_token)

        return target

    ################################################

    def build_model(self):
        self.relation_ground = AttHierarchicalGround(self.input_size, self.hidden_size, self.visual_dim, self.word_dim)

        self.relation_reconstruction = DecoderRNN(self.input_size, self.hidden_size, len(self.vocab), 1, 10)

        params = [{'params': self.relation_reconstruction.parameters()}, {'params': self.relation_ground.parameters()}]
        self.optimizer = torch.optim.Adam(params=params,
                                          lr=self.lr)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=2, verbose=True)

        self.relation_ground.to(self.device)
        self.relation_reconstruction.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def save_model(self, epoch):
        torch.save(self.relation_reconstruction.state_dict(),
                   osp.join(self.model_dir, '{}-transformer-reconstruct-{}.ckpt'.format(self.model_name,
                                                                                        "yys-stage2-epoch{}").format(
                       epoch)))
        print("reconstruct model saved at {}".format(osp.join(self.model_dir,
                                                              '{}-transformer-reconstruct-{}.ckpt'.format(
                                                                  self.model_name,
                                                                  "yys-stage2-epoch{}").format(epoch))))
        torch.save(self.relation_ground.state_dict(),
                   osp.join(self.model_dir, '{}-transformer-ground-{}.ckpt'.format(self.model_name,
                                                                                   "yys-stage2-epoch{}").format(
                       epoch)))
        print("ground model saved at {}".format(osp.join(self.model_dir,
                                                         '{}-transformer-ground-{}.ckpt'.format(self.model_name,
                                                                                                "yys-stage2-epoch{}").format(
                                                             epoch))))

    def resume(self):
        """
        Initialize with the basic model obtained at the 1st stage.
        :param epoch:
        :return:
        """
        # ground_model_file = osp.join(self.model_dir,'vRGV-basic-ground-3.ckpt')
        # reconstruct_model_file = osp.join(self.model_dir,'vRGV-basic-reconstruct-3.ckpt')

        # ground_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-ground-best-stage1-epoch1.ckpt"
        # reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-best-stage1-epoch1.ckpt"

        ground_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-ground-kl-warmup-stage2-epoch0.ckpt"
        reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-best-stage1-epoch1.ckpt"

        ground_dict = torch.load(ground_model_file)
        print("ground model loaded, at {}".format(ground_model_file))
        reconstruct_dict = torch.load(reconstruct_model_file)
        print("reconstruct model loaded, at {}".format(reconstruct_model_file))

        new_ground_dict = {}
        for k, v in self.relation_ground.state_dict().items():
            if k in ground_dict:
                v = ground_dict[k]
            new_ground_dict[k] = v
        self.relation_ground.load_state_dict(new_ground_dict)  # only reconstruction part is better

        new_reconstruct_dict = {}
        for k, v in self.relation_reconstruction.state_dict().items():
            if k in reconstruct_dict:
                v = reconstruct_dict[k]
            new_reconstruct_dict[k] = v

        self.relation_reconstruction.load_state_dict(new_reconstruct_dict)

    def run(self, pretrain=False):

        self.build_model()
        if pretrain:
            self.resume()
            print("pretrained model loaded(form ground_relation.py)")
        else:
            print("start training from scratch(from ground_relation.py)")
        # self.relation_reconstruction.position = self.relation_ground.position
        ################################
        # print("loading pretrained weights but the original weights do NOT include the Transformer Part")
        # self.relation_ground.load_state_dict(torch.load("/home/gpu4/Vigelos/vRGV_data/vRGV/models/vidvrd/visual_bbox_trans_temp2-ground-6.ckpt"),strict=False)
        # self.relation_reconstruction.load_state_dict(torch.load("/home/gpu4/Vigelos/vRGV_data/vRGV/models/vidvrd/visual_bbox_trans_temp2-reconstruct-6.ckpt"),strict=False)
        ################################

        save_loss = np.inf

        for epoch in range(0, self.epoch_num):
            train_loss = self.train(epoch)
            val_loss = self.val(epoch)

            print('==> Epoch:[{}/{}] [Training loss: {:.4f} {:.4f} Val loss: {:.4f} {:.4f}]'.
                  format(epoch, self.epoch_num, train_loss, np.exp(train_loss), val_loss, np.exp(val_loss)))

            self.scheduler.step(val_loss)

            if val_loss < save_loss:
                save_loss = val_loss
                self.save_model(epoch)

    def inspect(self):
        self.build_model()
        ground_model_file = "models/vidvrd/visual_bbox_trans_temp2-transformer-reconstruct-kl-warmup-rct-stage2-epoch4.ckpt"
        reconstruct_model_file = "/home/gpu4/Vigelos/vRGV/models/vidvrd/visual_bbox_trans_temp2-reconstruct-6.ckpt"

        ground_dict = torch.load(ground_model_file)
        print("ground model loaded, at {}".format(ground_model_file))
        reconstruct_dict = torch.load(reconstruct_model_file)
        print("reconstruct model loaded, at {}".format(reconstruct_model_file))

        new_ground_dict = {}
        for k, v in self.relation_ground.state_dict().items():
            if k in ground_dict:
                v = ground_dict[k]
            new_ground_dict[k] = v
        self.relation_ground.load_state_dict(new_ground_dict)  # only reconstruction part is better

        new_reconstruct_dict = {}
        for k, v in self.relation_reconstruction.state_dict().items():
            if k in reconstruct_dict:
                v = reconstruct_dict[k]
                new_reconstruct_dict[k] = v
        self.relation_reconstruction.load_state_dict(new_reconstruct_dict)
        print(self.val(0))

    def train(self, epoch):
        print('==> Epoch:[{}/{}] [training stage Encode_lr: {} Decode_lr: {}]'.
              format(epoch, self.epoch_num, self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[0]['lr']))

        self.relation_ground.train()
        self.relation_reconstruction.train()

        total_step = len(self.train_loader)
        epoch_loss = 0

        # bd_sub_attn = []
        # bd_obj_attn = []
        # bd_beta1 = []
        # bd_beta2 = []
        # bd_video_name = []
        # bd_relation_text = []

        for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.train_loader):

            videos = videos.to(self.device)
            relations = relations.to(self.device)

            # print("text:",len(relation_text))

            #################################
            # prepare the reverse relation
            reverse_relation_text = []
            video_num = []
            # filter all the bio-directional motions
            for i in range(len(relation_text)):
                motion = relation_text[i].split("-")[1]
                if motion in bd_motions:
                    video_num.append(i)
                    reverse_relation = "-".join(
                        [relation_text[i].split("-")[2], bd_motions[relation_text[i].split("-")[1]],
                         relation_text[i].split("-")[0]])
                    reverse_relation_text.append(reverse_relation)
            bdvideo = torch.zeros([len(video_num), 120, 40, 2053]).to(videos.device)
            reverse_valid_lengths = []
            for i in range(len(video_num)):
                bdvideo[i] = videos[video_num[i]]
                reverse_valid_lengths.append(valid_lengths[video_num[i]])

            reverse_video_out, reverse_video_hidden, reverse_sub_attn, reverse_obj_attn, reverse_beta1, reverse_beta2 = self.relation_ground(
                bdvideo, reverse_relation_text)
            ###################################

            targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

            video_out, video_hidden, sub_attn, obj_attn, normal_beta1, normal_beta2 = self.relation_ground(videos,
                                                                                                           relation_text)

            relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)

            ##########################################
            # filter all the sample of bd motions
            # bd_vid = []
            # for id in range(len(relation_text)):
            #     if relation_text[id].split("-")[1] in bd_motions:
            #         bd_vid.append(id)
            # bd_sub_attn.extend([sub_attn[vid] for vid in bd_vid])
            # bd_obj_attn.extend([obj_attn[vid] for vid in bd_vid])
            # bd_beta1.extend([normal_beta1[vid] for vid in bd_vid])
            # bd_beta2.extend([normal_beta2[vid] for vid in bd_vid])
            # bd_video_name.extend([video_names[vid] for vid in bd_vid])
            # bd_relation_text.extend([relation_text[vid] for vid in bd_vid])
            ##########################################

            # rct_loss_normal = self.criterion(relation_decode, targets)
            rct_loss = self.criterion(relation_decode, targets)

            #########################################################
            # add reverse relation to reconstruction loss
            # reverse_relation = [self.get_word_idx(subpredobj) for subpredobj in reverse_relation_text]
            # # merge relations
            # reverse_lengths = [len(rel) for rel in reverse_relation]
            # reverse_relation_token = torch.zeros(len(reverse_relation), max(reverse_lengths)).long().to(relations.device)
            # for i, rel in enumerate(reverse_relation):
            #     end = reverse_lengths[i]
            #     reverse_relation_token[i, :end] = rel[:end]
            # reverse_relation_decode = self.relation_reconstruction(reverse_video_out, reverse_video_hidden, reverse_relation_token, reverse_valid_lengths)
            # reverse_targets = pack_padded_sequence(reverse_relation_token, reverse_valid_lengths, batch_first=True)[0]
            # rct_loss_reverse = self.criterion(reverse_relation_decode, reverse_targets)
            # rct_loss = (rct_loss_normal + rct_loss_reverse)
            #########################################################

            ############################################
            # compute kl loss between normal and reverse video feature
            # normal_beta1 = torch.stack([normal_beta1[i] for i in video_num],dim=0)
            # normal_beta2 = torch.stack([normal_beta2[i] for i in video_num],dim=0)
            # normal_beta = torch.zeros_like(normal_beta1)
            # reverse_beta = torch.zeros_like(reverse_beta1)
            #
            # for i in range(120):
            #     normal_beta[:,i] = normal_beta1[:,i] + normal_beta2[:,i//12]
            #     reverse_beta[:,i] = reverse_beta1[:,i] + reverse_beta2[:,i//12]
            #
            # kl_loss = kl_div(normal_beta.log(), reverse_beta, reduction='sum')
            ################################################

            self.relation_ground.zero_grad()
            self.relation_reconstruction.zero_grad()

            # loss = (rct_loss+kl_loss+clr_loss)
            # loss = (rct_loss+kl_loss)
            # loss = kl_loss
            loss = rct_loss
            loss.backward()

            # clip_gradient(self.optimizer, self.grad_clip)
            # clip_grad_value_()
            self.optimizer.step()

            cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            if iter % self.vis_step == 0:
                print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
                      format(iter, total_step, cur_time, rct_loss.item(), np.exp(loss.item())))
                # print("    rct_loss:", rct_loss.item(), "kl_loss:", kl_loss.item(), "clr_loss:", clr_loss.item())
                # print("    rct_loss:", rct_loss.item(), "kl_loss:", kl_loss.item())
                # print("    kl_loss:", kl_loss.item())
                print("    rct_loss:", rct_loss.item())

        # data = [(bd_video_name[i],bd_relation_text[i],bd_sub_attn[i],bd_obj_attn[i],bd_beta1[i],bd_beta2[i]) for i in range(len(bd_video_name))]
        # with open('data-epoch-{}.pickle'.format(epoch), 'wb') as f:
        #     pickle.dump(data, f)

        return epoch_loss / total_step

    def val(self, epoch):
        print('==> Epoch:[{}/{}][validation stage]'.format(epoch, self.epoch_num))

        self.relation_ground.eval()
        self.relation_reconstruction.eval()

        total_step = len(self.val_loader)
        epoch_loss = 0

        correct_count = 0
        all_count = 0

        with torch.no_grad():
            for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):
                # print("relation text:",relation_text)
                # print("videos:",videos.shape)
                # print("relations token:",relations)
                # print("valid length:",valid_lengths)
                # print("video name:",video_names)
                videos = videos.to(self.device)
                relations = relations.to(self.device)
                targets = pack_padded_sequence(relations, valid_lengths, batch_first=True)[0]

                video_out, video_hidden, _, _a, _b, _c = self.relation_ground(videos, relation_text)
                relation_decode = self.relation_reconstruction(video_out, video_hidden, relations, valid_lengths)

                loss = self.criterion(relation_decode, targets)

                cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                if iter % self.vis_step == 0:
                    print('    [{}/{}]-{}-{:.4f}-{:5.4f}'.
                          format(iter, total_step, cur_time, loss.item(), np.exp(loss.item())))

                epoch_loss += loss.item()

        return epoch_loss / total_step

    def predict(self, ep):

        self.build_model()

        ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
        reconstruction_path = osp.join(self.model_dir, '{}-reconstruct-{}.ckpt'.format(self.model_name, ep))

        self.relation_reconstruction.eval()
        self.relation_ground.eval()

        self.relation_ground.load_state_dict(torch.load(ground_model_path))
        self.relation_reconstruction.load_state_dict(torch.load(reconstruction_path))
        total = len(self.val_loader)
        pos_num = 0

        fout = open('results/prediction.txt', 'w')
        with torch.no_grad():
            for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):

                videos = videos.to(self.device)

                video_out, video_hidden = self.relation_ground(videos, relation_text)

                sample_ids = self.relation_reconstruction.sample(video_out, video_hidden)

                sample_ids = sample_ids[0].cpu().numpy()

                predict_relation = []
                for id in sample_ids:
                    word = self.vocab.idx2word[id]
                    predict_relation.append(word)
                    if word == '<end>': break

                predict_relation = ' '.join(predict_relation[1:-1])
                # print(relation_text[0], predict_relation)

                table = str.maketrans('-_', '  ')
                relation = relation_text[0].translate(table)
                output_str = "{}:{}".format(relation, predict_relation)
                fout.writelines(output_str + '\n')

                if relation == predict_relation:
                    pos_num += 1

                if iter % self.vis_step == 0:
                    print("{}:{}".format(iter, output_str))

        print("Reconstrution Rate: ", pos_num / total)
        fout.close()

    def ground_attention(self, ep, save_name):
        """output the spatial temporal attention as grounding results"""
        self.build_model()
        # ground_model_path = osp.join(self.model_dir, '{}-ground-{}.ckpt'.format(self.model_name, ep))
        self.relation_ground.eval()
        # ground_model_path = "models/vidvrd/acc26.88.beta0008.only_aug.ckpt"
        ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-yys-stage2-epoch8.ckpt"
        # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-yys-stage2-epoch3.ckpt"
        # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-ground-6.ckpt"
        # ground_model_path = "models/vidvrd/visual_bbox_trans_temp2-transformer-ground-kl-warmup-rct-stage2-epoch7.ckpt"

        self.relation_ground.load_state_dict(torch.load(ground_model_path))
        print("load ground model from {}(from ground_relation.py)".format(ground_model_path))
        video_res = {}
        total = len(self.val_loader)

        with torch.no_grad():
            for iter, (relation_text, videos, relations, valid_lengths, video_names) in enumerate(self.val_loader):

                videos = videos.to(self.device)
                sub_atts, obj_atts, beta1, beta2 = self.relation_ground(videos, relation_text, mode='val')
                # print(sub_atts.shape, beta1.shape) #(32, 120, 40), (32, 120)

                data_sub_atts = sub_atts.data.cpu().numpy()
                data_obj_atts = obj_atts.data.cpu().numpy()
                data_beta2 = beta2.data.cpu().numpy()
                data_beta1 = beta1.data.cpu().numpy()
                real_bs = data_sub_atts.shape[0]

                for bs in range(real_bs):
                    data = {}
                    data['sub'] = data_sub_atts[bs].tolist()
                    data['obj'] = data_obj_atts[bs].tolist()
                    data['beta2'] = data_beta2[bs].tolist()
                    data['beta1'] = data_beta1[bs].tolist()

                    vname = video_names[bs]
                    if vname not in video_res: video_res[vname] = {}
                    video_res[vname][relation_text[bs]] = data

                if iter % self.vis_step == 0:
                    print('Finished: {}-{}'.format(iter, total))

            save_results(save_name, video_res)
