from collections import defaultdict
from sklearn.metrics import roc_auc_score
import numpy as np

def cal_group_auc(labels, preds, user_id_list):
    """
    Calculate group auc
    每个用户是一个组，计算每个用户的auc，∑每个用户的auc*权重 / ∑权重
    这里的权重用的是用户的曝光数量
    """
    print("*" * 50)
    if len(labels) != len(user_id_list):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda:[])
    group_truth = defaultdict(lambda:[])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    
    # 过滤掉单个用户全是正样本或负样本的情况
    group_flag = defaultdict(lambda:False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i+1]:
                flag = True
                break
        group_flag[user_id] = flag
    
    impression_total = 0
    total_auc = 0

    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]),np.asarray(group_score[user_id]))
            print("user_id:\t %s, auc:\t%s" % (user_id, auc))
            total_auc += auc * len(group_truth[user_id]) # ∑每个用户的auc*权重
            impression_total += len(group_truth[user_id]) # ∑权重
    group_auc = float(total_auc / impression_total)
    group_auc = round(group_auc, 4)
    return group_auc

if __name__ == '__main__':
    labels = [0,1,1,0,1]
    preds = [0.1,0.2,0.3,0.4,0.5]
    user_id_list = ['甲','甲','甲','乙','乙']
    group_auc = cal_group_auc(labels, preds, user_id_list)
    auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
    print("group_auc:%s, \t auc:%s" % (group_auc, auc))


# **************************************************
# user_id:     乙, auc:    1.0
# user_id:     甲, auc:    1.0
# group_auc:1.0,   auc:0.6666666666666666






    