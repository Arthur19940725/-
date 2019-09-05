import tensorflow as tf
import time
import os
import numpy as np
import imageio

Capsulemodel = __import__("8-2  Capsulemodel")
CapsuleNetModel = Capsulemodel.CapsuleNetModel

# 加载数据集

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./fashion/", one_hot= True)


def save_images(imgs, size, path): #定义函数，保存图片
    imgs = (imgs + 1.)/2
    return(imageio.imwrite(path, mergeImgs(imgs, size)))

def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h :j * h + h,i * w :i * w + w, :] = image
        imgs[j * h:j * h + h,i * w : i * w + w, :] = image
        return imgs

batch_size = 128
learning_rate = 1e-3
training_epochs = 5
n_class = 10
iter_routing = 3 #定义胶囊网络中动态路由训练次数


def main():
    capsmodel = CapsuleNetModel(batch_size, n_class, iter_routing)
    #实例化模型
    #
    capsmodel.building_model(is_train=True, learning_rate=learning_rate)
    os.makedirs('result', exist_ok=True)
    os.makedirs('./model', exist_ok=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #载入检查点文件
        checkpoint_path = tf.train.latest_checkpoint('./model/')
        print("点路径checkpoint_path is",checkpoint_path)
        if checkpoint_path != None:
            capsmodel.saver.restore(sess, checkpoint_path)
            history = []  # 此为收集loss值
            for epoch in range(training_epochs): # 按照指定迭代次数收集迭代数据集

                total_batch = int(mnist.train.num_examples/batch_size)
                lossvalue = 0
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)#取数据集
                    batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
                    tic = time.time()
                    _, loss_value = sess.run([capsmodel.train_op,capsmodel.total_loss],feed_dict={capsmodel.x : batch_x, capsmodel.y : batch_y})
                    loss_value += loss_value
                    if i % 20 == 0:
                        print(str(i)+ '用时: ' + str(time.time()-tic)+ 'loss:', loss_value)

                        cls_result, recon_imgs = sess.run([capsmodel.v_len, capsmodel.output], feed_dict={capsmodel.x : batch_x, capsmodel.y : batch_y})
                        imgs = np.reshape(recon_imgs, (batch_size, 28, 28 ,1))
                        size = 6
                        save_images(imgs[0:size * size,:], [size, size], 'results/test_%03d.png' %i) #将结果保存为图片
                        #获取分类结果，评估准确率
                        argmax_idx = np.argmax(cls_result,axis = 1)
                        batch_y_idx = np.argmax(batch_y,axis = 1)
                        cls_acc = np.mean(np.equal(argmax_idx,batch_y_idx).astype(np.float32))
                        print('正确率： ' + str(cls_acc * 100))
                        history.append(loss_value/total_batch) #保持本次迭代的loss值
                if lossvalue/total_batch == min(history): #如果loss值变小，保存模型
                    ckpt_path = os.path.join('./model','model.ckpt')
                    capsmodel.saver.save(sess, ckpt_path, global_step = capsmodel.global_step.eval() )#生成检点文件
                    print("save model", ckpt_path)

                print(epoch,lossvalue/total_batch)




if __name__ == "__main__":
tf.app.run()

