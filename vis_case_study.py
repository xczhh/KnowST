# -*- coding: utf-8 -*-
import Preprocess
import KnowST
import torch

save_path = r"saved_dict\0.9191.ckpt"

#attrs 27
#无attrs 18

word2id, word_embeddings, train_data, train_y, test_data, length_train, length_test = Preprocess.load_data_and_labels_fewshot()
model_config = KnowST.model_config()
model = KnowST.Model(word_embeddings, model_config)
test_iter = Preprocess.build_iterator(test_data, model_config)

model.load_state_dict(torch.load(save_path))
model.eval()

# texts = "肝脏 S5 、 S6 见 一 不规则形 团块状 异常 信号 影 ， 大小 约 109 mm × 46 mm × 101 mm ， 增强 扫描 动脉期 见 明显 强化 ， 门静脉期 及 延迟期 见 强化 明显 减低 ， 呈 “ 快进快出 ” 强化 模式 。 门脉 主干 及其 分支 显影 正常 。 肝内 、 外 胆管 未见 扩张 。 胆囊壁 不厚 。 胰腺 、 脾脏 形态 、 大小 未见 异常 。 胰管 无 扩张 ， 实质 内 未见 异常 信号 影 ， 未见 异常 强化 。 腹膜 后 未见 增大 的 淋巴结 ， 增强后 未见 异常 强化 灶 。"
# outputs, loss2, loss_cl, fusion_output, attr_end_pred, attr_early_pred = model(texts, "111", torch.tensor([[0, 0, 0, 1]]), [[1, 0, 0, 0, 0, 1, 0]], train=False)
with torch.no_grad():
    j=0
    for texts, labels, attrs_end, attrs_early1, attrs_early2 in test_iter:
        outputs, loss2, loss_cl, fusion_output, attr_end_pred, attr_early_pred1, attr_early_pred2, loss2_end_output, loss2_early_output = model(
texts, labels, attrs_end, attrs_early1, attrs_early2, train=False)
        j+=1
        for i in range(0, len(fusion_output)):
            print("**********************************")
            print("epoch: " + str(j) + "batch: " + str(i))
            print(outputs[i])
            print(fusion_output[i])
            # print(torch.max(attr_end_pred[i], 0)[1])
            # print(attr_end_pred[i])
            # print(torch.max(attr_early_pred1[i], 0)[1])
            # print(attr_early_pred1[i])
            # print(torch.max(attr_early_pred2[i], 0)[1])
            # print(attr_early_pred2[i])
            print("***********************************")

        print("batch")