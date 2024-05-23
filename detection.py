from mmdet.apis import DetInferencer

# pretrained_model: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
LABEL_MAP = {
    0: "cola",
    1: "pepsi",
    2: "sprite",
    3: "fanta",
    4: "sprint",
    5: "ice",
    6: "scream",
    7: "milk",
    8: "red",
    9: "king"
}
# 初始化模型
inferencer = DetInferencer(model='configs/rtmdet/rtmdet_tiny_1xb12-40e_drinks.py', 
                           weights='work_dir/best_coco_bbox_mAP_epoch_40.pth',
                           device='cuda:0')


def get_top_res(json_dict) -> list:
    res_list = []
    SCORE_THRESHOLD = 0.35
    
    labels = json_dict["labels"]
    scores = json_dict["scores"]
    for i in range(len(scores)):
        score = scores[i]
        if score > SCORE_THRESHOLD:
            res_list.append({LABEL_MAP[int(labels[i])]: score})
        else:
            break
    return res_list


def detect_one_img(img_path) -> list:
    infer = inferencer(img_path, out_dir='', no_save_vis=True, no_save_pred=True, print_result=False)
    preds = infer['predictions']
    res = []
    for pred in preds:
        res.extend(get_top_res(pred))
    return res


