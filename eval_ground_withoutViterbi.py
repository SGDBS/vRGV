import os.path as osp
from evaluations.common_witherViterbi import tiou
from evaluations.util import load_file
import generate_track_link
import pickle

static = ['above', 'beneath', 'left', 'right', 'front', 'behind', 'taller', 'larger', 'next_to', 'inside', 'hold', 'bite', 'lie_above', 'lie_beneath', 'lie_left', 'lie_right', 'lie_inside', 'lie_next_to', 'lie_with', 'stand_above', 'stand_beneath', 'stand_left', 'stand_right', 'stand_front', 'stand_behind', 'stand_next_to', 'stand_inside', 'sit_above', 'sit_left', 'sit_right', 'sit_front', 'sit_behind', 'sit_next_to', 'sit_inside', 'stop_above', 'stop_beneath', 'stop_left', 'stop_right', 'stop_front', 'stop_behind', 'stop_next_to', 'stop_with']

dynamic = ['swim_behind', 'walk_away', 'fly_behind', 'creep_behind', 'move_left', 'touch', 'follow', 'move_away', 'walk_with', 'move_next_to', 'creep_above', 'fall_off', 'run_with', 'swim_front', 'walk_next_to', 'kick', 'creep_right', 'watch', 'swim_with', 'fly_away', 'creep_beneath', 'run_past', 'jump_right', 'fly_toward', 'creep_left', 'run_next_to', 'jump_front', 'jump_beneath', 'past', 'jump_toward', 'walk_beneath', 'run_away', 'run_above', 'walk_right', 'away', 'move_right', 'fly_right', 'run_front', 'run_toward', 'jump_past', 'jump_above', 'move_with', 'swim_beneath', 'walk_past', 'run_right', 'creep_away', 'move_toward', 'feed', 'run_left', 'fly_front', 'walk_behind', 'fly_above', 'fly_next_to', 'fight', 'walk_above', 'jump_behind', 'fly_with', 'jump_next_to', 'run_behind', 'move_behind', 'swim_right', 'swim_next_to', 'move_past', 'pull', 'walk_left', 'ride', 'move_beneath', 'toward', 'jump_left', 'creep_toward', 'fly_left', 'walk_toward', 'chase', 'creep_next_to', 'fly_past', 'move_front', 'run_beneath', 'creep_front', 'creep_past', 'play', 'move_above', 'faster', 'walk_front', 'drive', 'swim_left', 'jump_away', 'jump_with']

# short_term = ['stand_behind', 'front', 'run_behind', 'bite', 'move_right', 'move_front', 'move_behind', 'move_left', 'move_beneath', 'sit_above', 'ride', 'behind', 'left', 'right', 'chase', 'faster', 'stand_above', 'move_toward', 'stand_left', 'move_past', 'stop_behind', 'walk_away', 'walk_behind', 'walk_right', 'stop_beneath', 'stop_front', 'stop_left', 'stand_front', 'stand_next_to', 'walk_next_to', 'walk_front', 'walk_left', 'walk_past', 'walk_toward', 'jump_right', 'jump_above', 'beneath', 'touch', 'run_front', 'run_away', 'follow', 'run_right', 'run_with', 'run_left', 'stop_right', 'creep_behind', 'creep_above', 'jump_toward', 'jump_left', 'jump_behind', 'jump_front', 'jump_beneath', 'past', 'sit_beneath', 'walk_above', 'next_to', 'run_beneath', 'stand_beneath', 'run_past', 'creep_front', 'fight', 'jump_past', 'stop_above', 'fly_next_to', 'fly_past', 'fly_front', 'fly_with', 'fly_behind', 'fly_right', 'fly_away', 'fly_toward', 'stand_inside', 'creep_left', 'creep_right', 'creep_next_to', 'creep_toward', 'run_next_to', 'away', 'jump_away', 'toward', 'run_above', 'fly_above', 'above', 'move_above', 'jump_next_to', 'jump_with', 'run_toward', 'stop_with', 'creep_past', 'swim_with', 'swim_right', 'swim_front', 'swim_next_to', 'swim_behind', 'creep_beneath', 'creep_away', 'kick', 'move_away']

# long_term = ['larger', 'watch', 'stand_with', 'stand_right', 'walk_with', 'play', 'taller', 'sit_behind', 'sit_right', 'lie_right', 'lie_next_to', 'lie_behind', 'lie_front', 'lie_beneath', 'lie_above', 'lie_left', 'sit_front', 'sit_next_to', 'sit_left', 'walk_beneath', 'lie_with', 'fly_left', 'hold', 'move_with', 'move_next_to', 'feed', 'swim_beneath', 'swim_left', 'sit_inside', 'drive', 'stop_next_to', 'pull', 'fall_off', 'lie_inside']

bd_motions = {'beneath': 'above', 'above': 'beneath', 'right': 'left', 'left': 'right', 'behind': 'front', 'front': 'behind', 'next_to': 'next_to', 'outside': 'inside', 'inside': 'outside', 'shorter': 'taller', 'taller': 'shorter', 'smaller': 'larger', 'larger': 'smaller', 'lie_beneath': 'lie_above', 'lie_above': 'lie_beneath', 'lie_right': 'lie_left', 'lie_left': 'lie_right', 'lie_outside': 'lie_inside', 'lie_inside': 'lie_outside', 'lie_next_to': 'lie_next_to', 'stand_beneath': 'stand_above', 'stand_above': 'stand_beneath', 'stand_right': 'stand_left', 'stand_left': 'stand_right', 'stand_behind': 'stand_front', 'stand_front': 'stand_behind', 'stand_next_to': 'stand_next_to', 'stand_outside': 'stand_inside', 'stand_inside': 'stand_outside', 'sit_beneath': 'sit_above', 'sit_above': 'sit_beneath', 'sit_right': 'sit_left', 'sit_left': 'sit_right', 'sit_behind': 'sit_front', 'sit_front': 'sit_behind', 'sit_next_to': 'sit_next_to', 'sit_outside': 'sit_inside', 'sit_inside': 'sit_outside'}

zero_shot = ['dog-run_behind-frisbee', 'dog-jump_above-sofa', 'person-stand_left-sofa', 'person-taller-sofa', 'sofa-right-person', 'sofa-larger-person', 'person-follow-dog', 'person-run_behind-dog', 'person-run_behind-dog', 'person-run_with-person', 'person-run_with-person', 'person-run_left-person', 'person-follow-dog', 'person-chase-dog', 'person-chase-dog', 'domestic_cat-creep_behind-ball', 'domestic_cat-creep_above-ball', 'ball-beneath-domestic_cat', 'red_panda-jump_left-red_panda', 'red_panda-jump_left-red_panda', 'red_panda-jump_behind-red_panda', 'giant_panda-jump_front-giant_panda', 'giant_panda-jump_toward-giant_panda', 'squirrel-sit_right-person', 'squirrel-stand_right-person', 'squirrel-play-person', 'person-left-squirrel', 'person-touch-squirrel', 'bicycle-jump_toward-person', 'giant_panda-next_to-giant_panda', 'giant_panda-walk_with-giant_panda', 'giant_panda-walk_with-giant_panda', 'giant_panda-front-giant_panda', 'giant_panda-faster-giant_panda', 'monkey-run_beneath-monkey', 'red_panda-run_past-red_panda', 'red_panda-run_beneath-red_panda', 'red_panda-run_right-red_panda', 'dog-lie_behind-snake', 'dog-lie_left-snake', 'dog-lie_behind-snake', 'dog-lie_left-snake', 'dog-run_behind-snake', 'dog-walk_toward-snake', 'dog-run_behind-snake', 'dog-run_away-snake', 'dog-run_behind-snake', 'snake-right-dog', 'dog-behind-snake', 'dog-behind-snake', 'cattle-right-cattle', 'antelope-jump_past-antelope', 'antelope-jump_behind-antelope', 'bicycle-stop_above-bicycle', 'bicycle-stop_beneath-bicycle', 'person-play-bicycle', 'turtle-lie_front-dog', 'turtle-lie_left-dog', 'dog-stand_behind-turtle', 'turtle-lie_right-dog', 'dog-stand_left-turtle', 'turtle-lie_front-dog', 'dog-stand_behind-turtle', 'dog-stand_left-turtle', 'cattle-sit_next_to-cattle', 'cattle-sit_front-cattle', 'cattle-sit_right-cattle', 'cattle-sit_left-cattle', 'bird-fly_next_to-bird', 'bird-fly_next_to-bird', 'bird-fly_toward-bird', 'bird-fly_toward-bird', 'person-stand_inside-car', 'person-stand_inside-car', 'bear-sit_right-bear', 'bear-sit_right-bear', 'bear-sit_right-bear', 'bear-creep_behind-bear', 'bear-creep_left-bear', 'bear-creep_behind-bear', 'bear-creep_next_to-bear', 'fox-lie_next_to-fox', 'bear-creep_behind-bear', 'bear-creep_toward-bear', 'dog-sit_right-fox', 'fox-sit_next_to-fox', 'fox-run_next_to-fox', 'fox-jump_away-fox', 'fox-jump_behind-fox', 'fox-jump_away-fox', 'fox-front-fox', 'fox-away-fox', 'fox-front-fox', 'fox-away-fox', 'person-hold-motorcycle', 'person-run_right-dog', 'bicycle-move_right-bird', 'bird-fly_left-bicycle', 'tiger-walk_next_to-tiger', 'tiger-walk_next_to-tiger', 'tiger-walk_right-tiger', 'tiger-walk_left-tiger', 'bird-chase-bicycle', 'bird-behind-person', 'bird-above-person', 'bird-left-person', 'bird-chase-person', 'person-front-bird', 'tiger-lie_right-tiger', 'tiger-lie_left-tiger', 'tiger-walk_right-tiger', 'tiger-walk_away-tiger', 'dog-sit_next_to-person', 'dog-sit_behind-person', 'person-front-dog', 'dog-sit_next_to-person', 'dog-sit_left-rabbit', 'rabbit-lie_right-dog', 'person-sit_left-rabbit', 'rabbit-lie_right-person', 'dog-lie_left-rabbit', 'person-taller-rabbit', 'person-larger-rabbit', 'person-follow-dog', 'monkey-sit_left-lizard', 'monkey-sit_left-lizard', 'monkey-sit_left-lizard', 'lizard-larger-monkey', 'lizard-larger-monkey', 'lizard-larger-monkey', 'antelope-jump_right-monkey', 'antelope-jump_away-antelope', 'antelope-jump_past-monkey', 'antelope-jump_front-monkey', 'watercraft-move_beneath-airplane', 'airplane-fly_above-watercraft', 'airplane-fly_toward-watercraft', 'watercraft-move_beneath-airplane', 'airplane-fly_past-watercraft', 'airplane-fly_above-watercraft', 'watercraft-move_beneath-airplane', 'airplane-move_above-watercraft', 'airplane-stop_above-watercraft', 'watercraft-larger-airplane', 'airplane-faster-watercraft', 'watercraft-larger-airplane', 'airplane-faster-watercraft', 'watercraft-larger-airplane', 'airplane-faster-airplane', 'person-feed-elephant', 'person-feed-elephant', 'bird-jump_next_to-bird', 'bird-jump_next_to-bird', 'bird-jump_next_to-bird', 'bird-next_to-bird', 'bird-next_to-bird', 'dog-stand_next_to-monkey', 'bear-swim_left-bear', 'dog-stand_front-monkey', 'dog-stand_left-monkey', 'dog-taller-monkey', 'monkey-sit_next_to-dog', 'dog-walk_next_to-monkey', 'monkey-jump_next_to-dog', 'dog-stand_next_to-monkey', 'monkey-jump_next_to-dog', 'monkey-jump_next_to-dog', 'monkey-sit_next_to-dog', 'monkey-stand_next_to-dog', 'dog-stand_next_to-monkey', 'monkey-sit_right-dog', 'monkey-sit_behind-dog', 'dog-walk_front-monkey', 'dog-walk_left-monkey', 'monkey-next_to-dog', 'monkey-jump_right-dog', 'dog-stand_left-monkey', 'monkey-next_to-dog', 'monkey-jump_right-dog', 'monkey-next_to-dog', 'monkey-jump_right-dog', 'monkey-sit_right-dog', 'monkey-stand_right-dog', 'dog-larger-monkey', 'dog-watch-monkey', 'monkey-watch-dog', 'dog-play-monkey', 'monkey-play-dog', 'monkey-right-dog', 'monkey-touch-dog', 'monkey-right-dog', 'monkey-right-dog', 'dog-watch-monkey', 'monkey-jump_next_to-dog', 'monkey-sit_next_to-dog', 'monkey-run_next_to-dog', 'dog-stand_left-monkey', 'monkey-jump_right-dog', 'monkey-next_to-dog', 'monkey-sit_right-dog', 'monkey-run_away-dog', 'monkey-run_right-dog', 'monkey-run_past-person', 'monkey-run_left-person', 'dog-taller-monkey', 'dog-larger-monkey', 'dog-play-monkey', 'dog-bite-monkey', 'dog-watch-monkey', 'monkey-play-dog', 'monkey-watch-dog', 'monkey-touch-dog', 'monkey-right-dog', 'dog-watch-monkey', 'domestic_cat-lie_left-bird', 'bird-stand_right-domestic_cat', 'fox-walk_with-fox', 'fox-walk_with-fox', 'person-feed-elephant', 'person-feed-elephant', 'car-stop_with-car', 'car-stop_with-car', 'monkey-lie_behind-monkey', 'monkey-creep_left-monkey', 'dog-sit_behind-person', 'lizard-lie_behind-lizard', 'lizard-creep_left-lizard', 'lizard-creep_left-lizard', 'lizard-lie_behind-lizard', 'lizard-creep_left-lizard', 'lizard-creep_left-lizard', 'lizard-lie_behind-lizard', 'lizard-lie_behind-lizard', 'lizard-lie_behind-lizard', 'lizard-lie_behind-lizard', 'lizard-creep_right-lizard', 'lizard-creep_right-lizard', 'lizard-creep_left-lizard', 'lizard-creep_left-lizard', 'lizard-creep_right-lizard', 'lizard-creep_past-lizard', 'lizard-creep_left-lizard', 'lizard-creep_past-lizard', 'lizard-creep_left-lizard', 'lizard-creep_right-lizard', 'lizard-creep_right-lizard', 'lizard-lie_behind-lizard', 'lizard-lie_behind-lizard', 'lizard-lie_behind-lizard', 'lizard-creep_right-lizard', 'lizard-creep_left-lizard', 'lizard-lie_behind-lizard', 'lizard-creep_right-lizard', 'lizard-creep_right-lizard', 'lizard-lie_behind-lizard', 'lizard-creep_left-lizard', 'lizard-lie_behind-lizard', 'rabbit-lie_front-car', 'car-stop_behind-rabbit', 'rabbit-lie_front-car', 'rabbit-lie_left-car', 'car-stop_right-rabbit', 'car-stop_behind-rabbit', 'rabbit-sit_front-car', 'rabbit-sit_left-car', 'watercraft-swim_behind-whale', 'watercraft-swim_behind-whale', 'watercraft-swim_behind-whale', 'watercraft-swim_left-whale', 'watercraft-stop_beneath-person', 'person-walk_toward-whale', 'person-walk_right-whale', 'watercraft-stop_beneath-person', 'person-walk_above-watercraft', 'bird-jump_beneath-bird', 'bird-fight-bird', 'bird-fight-bird', 'bird-walk_beneath-bird', 'bird-walk_beneath-bird', 'bird-beneath-bird', 'watercraft-move_right-airplane', 'watercraft-move_behind-airplane', 'airplane-move_front-watercraft', 'airplane-move_left-watercraft', 'airplane-move_past-watercraft', 'dog-sit_behind-person', 'red_panda-jump_above-red_panda', 'dog-run_behind-person', 'red_panda-jump_left-red_panda', 'red_panda-jump_behind-red_panda', 'red_panda-jump_left-red_panda', 'lion-touch-lion', 'monkey-creep_left-monkey', 'monkey-creep_next_to-monkey', 'monkey-creep_above-monkey', 'monkey-creep_front-monkey', 'monkey-creep_left-monkey', 'monkey-creep_front-monkey', 'monkey-jump_next_to-monkey', 'monkey-creep_beneath-monkey', 'monkey-creep_left-monkey', 'monkey-creep_beneath-monkey', 'monkey-creep_beneath-monkey', 'monkey-creep_left-monkey', 'monkey-creep_beneath-monkey', 'monkey-creep_left-monkey', 'monkey-lie_next_to-monkey', 'monkey-lie_behind-monkey', 'monkey-lie_left-monkey', 'monkey-lie_left-monkey', 'monkey-lie_next_to-monkey', 'monkey-lie_next_to-monkey', 'monkey-lie_next_to-monkey', 'monkey-lie_behind-monkey', 'monkey-lie_left-monkey', 'monkey-lie_behind-monkey', 'monkey-lie_left-monkey', 'bicycle-lie_behind-car', 'bicycle-lie_left-car', 'bicycle-lie_left-person', 'person-lie_right-bicycle', 'person-lie_front-car', 'person-lie_left-car', 'person-lie_behind-car', 'person-above-bicycle', 'lizard-creep_above-domestic_cat', 'lizard-creep_away-domestic_cat', 'domestic_cat-sit_beneath-lizard', 'lizard-lie_above-domestic_cat', 'domestic_cat-walk_beneath-lizard', 'domestic_cat-stand_beneath-lizard', 'domestic_cat-beneath-lizard', 'domestic_cat-taller-lizard', 'domestic_cat-larger-lizard', 'domestic_cat-watch-lizard', 'domestic_cat-fight-lizard', 'lizard-above-domestic_cat', 'lizard-creep_above-domestic_cat', 'domestic_cat-walk_beneath-lizard', 'domestic_cat-walk_away-lizard', 'domestic_cat-walk_right-lizard', 'lizard-lie_above-domestic_cat', 'lizard-lie_left-domestic_cat', 'domestic_cat-stand_beneath-lizard', 'domestic_cat-beneath-lizard', 'domestic_cat-watch-lizard', 'dog-sit_next_to-dog', 'dog-sit_right-dog', 'person-above-bicycle', 'zebra-run_toward-zebra', 'zebra-run_toward-zebra', 'zebra-run_toward-zebra', 'zebra-chase-zebra', 'zebra-run_next_to-zebra', 'zebra-run_toward-zebra', 'zebra-chase-zebra', 'zebra-chase-zebra', 'zebra-run_next_to-zebra', 'zebra-run_next_to-zebra', 'zebra-run_toward-zebra', 'zebra-run_toward-zebra', 'zebra-run_toward-zebra', 'zebra-chase-zebra', 'zebra-run_next_to-zebra', 'zebra-chase-zebra', 'zebra-run_next_to-zebra', 'skateboard-move_behind-person', 'person-walk_right-skateboard', 'person-walk_front-skateboard', 'person-run_front-dog', 'person-run_front-skateboard', 'person-run_left-dog', 'person-run_left-skateboard', 'horse-stand_behind-person', 'elephant-stand_right-ball', 'elephant-taller-ball', 'elephant-touch-ball', 'elephant-kick-ball', 'ball-left-elephant', 'elephant-stand_behind-ball', 'elephant-stand_left-ball', 'elephant-play-person', 'elephant-watch-person', 'elephant-touch-person', 'person-play-elephant', 'elephant-taller-ball', 'ball-front-elephant', 'elephant-stand_behind-ball', 'elephant-stand_left-ball', 'person-walk_behind-ball', 'person-walk_away-ball', 'elephant-taller-ball', 'ball-front-elephant', 'ball-front-person', 'person-stand_behind-ball', 'person-stand_left-ball', 'person-stand_behind-ball', 'person-stand_left-ball', 'ball-front-person', 'ball-front-person', 'elephant-right-person', 'elephant-right-person', 'elephant-right-person', 'elephant-right-ball', 'ball-away-elephant', 'person-stand_behind-ball', 'person-taller-ball', 'person-feed-elephant', 'person-feed-elephant', 'elephant-lie_behind-elephant', 'elephant-run_right-person', 'elephant-run_right-person', 'elephant-run_behind-person', 'elephant-stand_above-elephant', 'elephant-run_beneath-elephant', 'elephant-walk_beneath-elephant', 'elephant-stand_beneath-elephant', 'whale-jump_next_to-whale', 'whale-jump_right-whale', 'whale-jump_left-whale', 'whale-jump_right-whale', 'whale-jump_right-watercraft', 'giant_panda-next_to-giant_panda', 'giant_panda-behind-giant_panda', 'person-lie_next_to-dog', 'dog-jump_next_to-person', 'dog-sit_next_to-person', 'bicycle-move_toward-bicycle', 'bicycle-move_toward-bicycle', 'person-toward-bicycle', 'person-toward-bicycle', 'squirrel-lie_inside-person']

no_static_video = 0
no_dynamic_video = 0
no_zero_shot_video = 0
# no_long_video = 0
# no_short_video = 0

def eval_ground_scores(qid, gt_relations, pred_relations, tiou_threshold):
    """
    :param gt_relations:
    :param pred_relations:
    :param tiou_threshold:
    :return:
    """
    # pred_relations = sorted(pred_relations, key=lambda x: x['score'], reverse=True)

    relation_num = len(gt_relations)
    relation_num_static, relation_num_dynamic, relation_num_zero_shot = 0, 0, 0
    # relation_num_long, relation_num_short = 0,0
    predict, predict_sub, predict_obj = 0, 0, 0
    reverse_predict, reverse_predict_sub, reverse_predict_obj = 0, 0, 0
    predict_static,predict_sub_static,predict_obj_static = 0,0,0
    predict_dynamic,predict_sub_dynamic,predict_obj_dynamic = 0,0,0
    predict_zero_shot,predict_sub_zero_shot,predict_obj_zero_shot = 0,0,0
    # predict_long,predict_sub_long,predict_obj_long = 0,0,0
    # predict_short,predict_sub_short,predict_obj_short = 0,0,0
    global no_static_video,no_dynamic_video, no_zero_shot_video #,no_long_video,no_short_video

    bd_motions_correct = 0
    bd_motions_all = 0

    acc_temporal = 0
    acc_spatial = 0
    temporal_count = 0
    spatial_count = 0


    for relation, pred_trajs in pred_relations.items():
        pred_sub = pred_trajs['sub']
        pred_obj = pred_trajs['obj']

        flag, flag_s, flag_o = False, False, False
        reverse_flag,reverse_flag_s,reverse_flag_o = False,False,False

        gt_trajs = gt_relations[relation]

        verb = relation.split("-")[1]

        if verb in bd_motions:
            bd_motions_all += 1

        temporal_count += len(gt_trajs)
        spatial_count += len(gt_trajs)
        for gt_traj in gt_trajs:
            gt_sub = gt_traj['sub']
            gt_obj = gt_traj['obj']

            ############################################
            # compute the temporal accuracy
            gt_temporal = list(gt_traj['sub'].keys())
            pred_temporal = list(pred_sub.keys())
            common = len(set(pred_temporal).intersection(gt_temporal))
            union = len(pred_temporal) + len(gt_temporal) - common
            acc_temporal += common * 1.0 / union
            ###########################################

            if common==0:
                # if there is no common temporal segment, then the spatial accuracy is 0
                spatial_count-=1
                s_tiou,o_tiou = 0,0
            else:
                # compute the spatial accuracy
                s_tacc, s_tiou = tiou(pred_sub, gt_sub)
                o_tacc, o_tiou = tiou(pred_obj, gt_obj)
                acc_spatial += (s_tacc + o_tacc) * 1.0 / 2

            r_iou = min(s_tiou, o_tiou)
            if r_iou >= tiou_threshold:
                flag = True
            if s_tiou >= tiou_threshold:
                flag_s = True
            if o_tiou >= tiou_threshold:
                flag_o = True

        # check the reverse order accuracy( exchange the order of sub and obj)
        for gt_traj in gt_trajs:
            gt_sub = gt_traj['obj']
            gt_obj = gt_traj['sub']

            s_tacc, s_tiou = tiou(pred_sub, gt_sub)
            o_tacc, o_tiou = tiou(pred_obj, gt_obj)

            reverse_r_iou = min(s_tiou,o_tiou)
            if reverse_r_iou >= tiou_threshold:
                reverse_flag = True
            if s_tiou >= tiou_threshold:
                reverse_flag_s = True
            if o_tiou >= tiou_threshold:
                reverse_flag_o = True

        if flag:
            predict += 1
            if verb in bd_motions:
                bd_motions_correct += 1
        if flag_s:
            predict_sub += 1
        if flag_o:
            predict_obj += 1

        if reverse_flag:
            reverse_predict += 1
        if reverse_flag_s:
            reverse_predict_sub += 1
        if reverse_flag_o:
            reverse_predict_obj += 1

        if verb in static:
            relation_num_static+=1
            if flag:
                predict_static += 1
            if flag_s:
                predict_sub_static += 1
            if flag_o:
                predict_obj_static += 1
        elif verb in dynamic:
            relation_num_dynamic+=1
            if flag:
                predict_dynamic += 1
            if flag_s:
                predict_sub_dynamic += 1
            if flag_o:
                predict_obj_dynamic += 1

        if relation in zero_shot:
            relation_num_zero_shot+=1
            if flag:
                predict_zero_shot += 1
            if flag_s:
                predict_sub_zero_shot += 1
            if flag_o:
                predict_obj_zero_shot += 1

        # if verb in long_term:
        #     relation_num_long+=1
        #     if flag:
        #         predict_long += 1
        #     if flag_s:
        #         predict_sub_long += 1
        #     if flag_o:
        #         predict_obj_long += 1
        # else:
        #     relation_num_short+=1
        #     if flag:
        #         predict_short += 1
        #     if flag_s:
        #         predict_sub_short += 1
        #     if flag_o:
        #         predict_obj_short += 1

    predict = predict / relation_num
    predict_sub = predict_sub /relation_num
    predict_obj = predict_obj /relation_num
    reverse_predict = reverse_predict / relation_num
    reverse_predict_sub = reverse_predict_sub /relation_num
    reverse_predict_obj = reverse_predict_obj /relation_num
    acc_temporal = acc_temporal / temporal_count

    if(acc_spatial==0):
        pass
    else:
        acc_spatial = acc_spatial / spatial_count

    if(relation_num_static!=0):
        predict_static = predict_static / relation_num_static
        predict_sub_static = predict_sub_static / relation_num_static
        predict_obj_static = predict_obj_static / relation_num_static
    else:
        no_static_video+=1
    if(relation_num_dynamic!=0):
        predict_dynamic = predict_dynamic / relation_num_dynamic
        predict_sub_dynamic = predict_sub_dynamic / relation_num_dynamic
        predict_obj_dynamic = predict_obj_dynamic / relation_num_dynamic
    else:
        no_dynamic_video+=1
    if(relation_num_zero_shot!=0):
        predict_zero_shot = predict_zero_shot / relation_num_zero_shot
        predict_obj_zero_shot = predict_obj_zero_shot / relation_num_zero_shot
        predict_sub_zero_shot = predict_sub_zero_shot / relation_num_zero_shot
    else:
        no_zero_shot_video+=1
    # if(relation_num_long!=0):
    #     predict_long = predict_long / relation_num_long
    #     predict_sub_long = predict_sub_long / relation_num_long
    #     predict_obj_long = predict_obj_long / relation_num_long
    # else:
    #     no_long_video+=1
    # if(relation_num_short!=0):
    #     predict_short = predict_short / relation_num_short
    #     predict_sub_short = predict_sub_short / relation_num_short
    #     predict_obj_short = predict_obj_short / relation_num_short
    # else:
    #     no_short_video+=1

    # return acc_spatial,acc_temporal,bd_motions_all,bd_motions_correct,predict, predict_sub, predict_obj, predict_static, predict_sub_static, predict_obj_static, predict_dynamic, predict_sub_dynamic, predict_obj_dynamic, predict_long, predict_sub_long, predict_obj_long, predict_short, predict_sub_short, predict_obj_short
    return reverse_predict,reverse_predict_sub,reverse_predict_obj,acc_spatial,acc_temporal,bd_motions_all,bd_motions_correct,predict, predict_sub, predict_obj, predict_static, predict_sub_static, predict_obj_static, predict_dynamic, predict_sub_dynamic, predict_obj_dynamic, predict_zero_shot, predict_sub_zero_shot, predict_obj_zero_shot


def evaluate(groundtruth, prediction, tiou_threshold=0.5):
    """ evaluate visual relation detection and visual
    relation tagging.
    """
    global no_static_video,no_dynamic_video,no_zero_shot_video #,no_long_video,no_short_video
    video_num = len(groundtruth)
    count = video_num
    print('Computing grounding accuracy over {} videos...'.format(video_num))
    with open('readme.txt', 'a') as f:
            f.write('Computing grounding accuracy over {} videos...'.format(video_num))
    acc, acc_sub, acc_obj = 0.0, 0.0, 0.0
    acc_reverse, acc_sub_reverse, acc_obj_reverse = 0.0, 0.0, 0.0
    acc_static, acc_sub_static, acc_obj_static = 0.0,0.0,0.0
    acc_dynamic, acc_sub_dynamic, acc_obj_dynamic = 0.0, 0.0, 0.0
    acc_zero_shot, acc_sub_zero_shot, acc_obj_zero_shot = 0.0, 0.0, 0.0
    # acc_long, acc_sub_long, acc_obj_long = 0.0,0.0,0.0
    # acc_short, acc_sub_short, acc_obj_short = 0.0, 0.0, 0.0
    bd_motions_correct = 0
    bd_motions_all = 0

    gt_rnum = 0
    acc_temporal=0
    acc_spatial=0

    for qid, relation_gt in groundtruth.items():
        if qid not in prediction:
            print('Warning: video {} missing in prediction. '.format(qid))
            continue
        relation_pred = prediction[qid]


        if len(relation_pred) == 0:
            continue

        # video_acc_spatial, video_acc_temporal, video_bd_motions_all, video_bd_motions_correct, video_acc, video_acc_sub, video_acc_obj, video_acc_static, video_acc_sub_static, video_acc_obj_static, video_acc_dynamic, video_acc_sub_dynamic, video_acc_obj_dynamic, video_acc_long, video_acc_sub_long, video_acc_obj_long, video_acc_short, video_acc_sub_short, video_acc_obj_short =  eval_ground_scores(relation_gt, relation_pred, tiou_threshold)
        video_acc_reverse, video_acc_sub_reverse, video_acc_obj_reverse, video_acc_spatial, video_acc_temporal, video_bd_motions_all, video_bd_motions_correct, video_acc, video_acc_sub, video_acc_obj, video_acc_static, video_acc_sub_static, video_acc_obj_static, video_acc_dynamic, video_acc_sub_dynamic, video_acc_obj_dynamic, video_acc_zero_shot, video_acc_sub_zero_shot, video_acc_obj_zero_shot =  eval_ground_scores(qid, relation_gt, relation_pred, tiou_threshold)

        acc_temporal += video_acc_temporal
        if acc_spatial==0:
            count-=1
        acc_spatial += video_acc_spatial

        acc += video_acc
        acc_sub += video_acc_sub
        acc_obj += video_acc_obj
        acc_reverse += video_acc_reverse
        acc_sub_reverse += video_acc_sub_reverse
        acc_obj_reverse += video_acc_obj_reverse
        #gt_rnum += relation_num

        acc_static += video_acc_static
        acc_sub_static += video_acc_sub_static
        acc_obj_static += video_acc_obj_static

        acc_dynamic += video_acc_dynamic
        acc_sub_dynamic += video_acc_sub_dynamic
        acc_obj_dynamic += video_acc_obj_dynamic

        acc_zero_shot += video_acc_zero_shot
        acc_sub_zero_shot += video_acc_sub_zero_shot
        acc_obj_zero_shot += video_acc_obj_zero_shot

        # acc_long += video_acc_long
        # acc_sub_long += video_acc_sub_long
        # acc_obj_long += video_acc_obj_long
        #
        # acc_short += video_acc_short
        # acc_sub_short += video_acc_sub_short
        # acc_obj_short += video_acc_obj_short

        bd_motions_all+=video_bd_motions_all
        bd_motions_correct+=video_bd_motions_correct


    acc /= video_num
    acc_sub /= video_num
    acc_obj /= video_num
    acc_reverse /= video_num
    acc_sub_reverse /= video_num
    acc_obj_reverse /= video_num

    acc_static /= (video_num-no_static_video)
    acc_sub_static /= (video_num-no_static_video)
    acc_obj_static /= (video_num-no_static_video)
    acc_dynamic /= (video_num-no_dynamic_video)
    acc_sub_dynamic /= (video_num-no_dynamic_video)
    acc_obj_dynamic /= (video_num-no_dynamic_video)
    acc_zero_shot /= (video_num-no_zero_shot_video)
    acc_sub_zero_shot /= (video_num-no_zero_shot_video)
    acc_obj_zero_shot /= (video_num-no_zero_shot_video)

    # acc_long /= (video_num-no_long_video)
    # acc_sub_long /= (video_num-no_long_video)
    # acc_obj_long /= (video_num-no_long_video)
    # acc_short /= (video_num-no_short_video)
    # acc_sub_short /= (video_num-no_short_video)
    # acc_obj_short /= (video_num-no_short_video)

    acc_bd = bd_motions_correct/bd_motions_all
    acc_temporal = acc_temporal/video_num
    acc_spatial = acc_spatial/count

    print("Overall ACC")
    print("Acc_S\t Acc_O\t Acc_R")
    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub*100, acc_obj*100, acc*100))

    print("reverse Overall ACC")
    print("Acc_S\t Acc_O\t Acc_R")
    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_reverse*100, acc_obj_reverse*100, acc_reverse*100))

    print("bd motions ACC:")
    print("{:.2f}".format(acc_bd*100))

    print("acc_temporal:")
    print("{:.2f}".format(acc_temporal*100))

    print("acc_spatial:")
    print("{:.2f}".format(acc_spatial*100))

    print("Static ACC")
    print("Acc_S\t Acc_O\t Acc_R")
    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_static*100, acc_obj_static*100, acc_static*100))

    print("Dynamic ACC")
    print("Acc_S\t Acc_O\t Acc_R")
    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_dynamic*100, acc_obj_dynamic*100, acc_dynamic*100))


    print("Zero-shot ACC")
    print("Acc_S\t Acc_O\t Acc_R")
    print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_zero_shot*100, acc_obj_zero_shot*100, acc_zero_shot*100))
    #
    # print("Long-term Motion ACC")
    # print("Acc_S\t Acc_O\t Acc_R")
    # print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_long*100, acc_obj_long*100, acc_long*100))
    #
    # print("Short-term Motion ACC")
    # print("Acc_S\t Acc_O\t Acc_R")
    # print('{:.2f}\t {:.2f}\t {:.2f}'.format(acc_sub_short*100, acc_obj_short*100, acc_short*100))




def main():

    groundtruth_dir = 'dataset/vidvrd/'
    gt_file = osp.join(groundtruth_dir, 'gt_relation_frame.json')

    result_dir = 'results/'
    res_file = osp.join(result_dir, 'test_viterbi_1gap_04_batch_without_viterbi.json')
    res_file_1 = osp.join(result_dir, 'test_viterbi_1gap_04_batch.json')
    if not osp.exists(res_file):
        print('Generating ...')
        generate_track_link.main(res_file)
    print("load gt file:",gt_file)
    print("load res file:",res_file)
    grountruth = load_file(gt_file)
    prediction = load_file(res_file)
    prediction_1 = load_file(res_file_1)


    for i in prediction.keys():
        for motion in prediction[i].keys():
            print("motion:",motion)
            print("    without vtb length:",prediction[i][motion]['sub'].keys())
            print("    normal length:",prediction_1[i][motion]['sub'].keys())
        exit(0)
    # evaluate(grountruth, prediction)



if __name__ == "__main__":

    main()
