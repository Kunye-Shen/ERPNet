import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import ERPNet

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')

def muti_label_loss(l0,l1,l2,l3,l4,l5,labels_v):
    loss0 = bce_loss(l0,labels_v)
    loss1 = bce_loss(l1,labels_v)
    loss2 = bce_loss(l2,labels_v)
    loss3 = bce_loss(l3,labels_v)
    loss4 = bce_loss(l4,labels_v)
    loss5 = bce_loss(l5,labels_v)
    loss_label = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    return loss_label

def muti_edge_loss(e1,e2,e3,e4,e5,edges_v):
    loss1 = bce_loss(e1,edges_v)
    loss2 = bce_loss(e2,edges_v)
    loss3 = bce_loss(e3,edges_v)
    loss4 = bce_loss(e4,edges_v)
    loss5 = bce_loss(e5,edges_v)

    loss_edge = loss1 + loss2 + loss3 + loss4 + loss5
    return loss_edge

# ------- 2. set the directory of training dataset --------

tra_image_dir = ""
tra_label_dir = ""
tra_edge_dir= ""

image_ext = '.jpg'
label_ext = '.png'
edge_ext = '.png'

model_dir = ""

epoch_num = 91
batch_size_train = 6
train_num = 0

tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
tra_edge_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split("/")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)
    tra_edge_name_list.append(tra_edge_dir + imidx + edge_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train edges:  ",len(tra_edge_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    edge_name_list=tra_edge_name_list,
    transform=transforms.Compose([
        RescaleT(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

# ------- 3. define model --------
# define the net
net = ERPNet()
if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 6. training process --------
def main():
    print("---start training...")
    ite_num = 0

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1

            inputs, labels, edges = data['image'], data['label'], data['edge']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            edges = edges.type(torch.FloatTensor)

        # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False)
                edges_v = Variable(edges.cuda(), requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
                edges_v = Variable(edges, requires_grad=False)

        # y zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            l0,l1,l2,l3,l4,l5,e1,e2,e3,e4,e5 = net(inputs_v)
            loss_label = muti_label_loss(l0,l1,l2,l3,l4,l5,labels_v)
            loss_edge = muti_edge_loss(e1,e2,e3,e4,e5,edges_v)
            loss = loss_label + loss_edge
            
            loss.backward()
            optimizer.step()

        # del temporary outputs and loss
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %.3f" % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, loss.item()))

            if ite_num % 2000 == 0:  # save model every 2000 iterations
                torch.save(net.state_dict(), model_dir + "MYNet_%d_%d.pth" % (ite_num, epoch))
                net.train()  # resume train

            del l0,l1,l2,l3,l4,l5,e1,e2,e3,e4,e5,loss_label,loss_edge,loss

    print('-------------Congratulations! Training Done!!!-------------')
if __name__ == '__main__':
    main()