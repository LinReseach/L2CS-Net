import os
import argparse
import time

import pandas as pd
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import datasets_local_lin as datasets_local
# import datasets
from model import L2CS
from utils_local import select_device, poly_lr_scheduler
#from utils import select_device, poly_lr_scheduler

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
    # socialai
    parser.add_argument(
        '--gazeSocialAI', dest='gazeSocialAImage_dir', help='Directory path for gaze images.',
        default='datasets/SocialAI/', type=str)
    parser.add_argument(
        '--gazeSocialAIlabel_dir', dest='gazeSocialAIlabel_dir', help='Directory path for gaze labels.',
        default='datasets/SocialAI/', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze, socialai',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='output/snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='snapshot/latest_model.pkl', type=str)
    parser.add_argument(
        '--validation_step', dest='validation_step', help='',
        default='1', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=32, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def getArch_weights(arch, bins):
    if arch == 'ResNet18':
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        pre_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    elif arch == 'ResNet34':
        model = L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
    elif arch == 'ResNet101':
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    elif arch == 'ResNet152':
        model = L2CS(torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                  'The default value of ResNet50 will be used instead!')
        model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url


if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set = args.dataset
    alpha = args.alpha
    output = args.output
    validation_step = args.validation_step
    
    
    transformations = transforms.Compose([
        transforms.Resize((448, 448)),  # old: transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    
    
    if data_set=="gaze360":
        model, pre_url = getArch_weights(args.arch, 90)
        if args.snapshot == '':
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        else:
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)
        
        
        model.cuda(gpu)
        dataset=datasets_local.Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        
        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)
        

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model), 'lr': 0},
            {'params': get_non_ignored_params(model), 'lr': args.lr},
            {'params': get_fc_params(model), 'lr': args.lr}
        ], args.lr)
       

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
            print("epoch",1)
            
            for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                images_gaze = Variable(images_gaze).cuda(gpu)
                
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                
                iter_gaze += 1

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset)//batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )
        
          
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )


    elif data_set=="mpiigaze":
        folder = os.listdir(args.gazeMpiilabel_dir)
        folder.sort()
        testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder]
        for fold in range(15):
            model, pre_url = getArch_weights(args.arch, 28)
            load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            model = nn.DataParallel(model)
            model.to(gpu)
            print('Loading data.')
            dataset=datasets_local.Mpiigaze(testlabelpathombined,args.gazeMpiimage_dir, transformations, True, fold)
            train_loader_gaze = DataLoader(
                dataset=dataset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            torch.backends.cudnn.benchmark = True

            summary_name = '{}_{}'.format('L2CS-mpiigaze', int(time.time()))
            

            if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
                os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))

            
            criterion = nn.CrossEntropyLoss().cuda(gpu)
            reg_criterion = nn.MSELoss().cuda(gpu)
            softmax = nn.Softmax(dim=1).cuda(gpu)
            idx_tensor = [idx for idx in range(28)]
            idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

            # Optimizer gaze
            optimizer_gaze = torch.optim.Adam([
                {'params': get_ignored_params(model, args.arch), 'lr': 0},
                {'params': get_non_ignored_params(model, args.arch), 'lr': args.lr},
                {'params': get_fc_params(model, args.arch), 'lr': args.lr}
            ], args.lr)

            

            configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\n Start training dataset={data_set}, loader={len(train_loader_gaze)}, fold={fold}--------------\n"
            print(configuration)
            for epoch in range(num_epochs):
                sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

                
                for i, (images_gaze, labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):
                    images_gaze = Variable(images_gaze).cuda(gpu)

                    # Binned labels
                    label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)

                    # Continuous labels
                    label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                    label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                    pitch, yaw = model(images_gaze)

                    # Cross entropy loss
                    loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                    loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                    # MSE loss
                    pitch_predicted = softmax(pitch)
                    yaw_predicted = softmax(yaw)
                    pitch_predicted = \
                        torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 42
                    yaw_predicted = \
                        torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42

                    loss_reg_pitch = reg_criterion(
                        pitch_predicted, label_pitch_cont_gaze)
                    loss_reg_yaw = reg_criterion(
                        yaw_predicted, label_yaw_cont_gaze)

                    # Total loss
                    loss_pitch_gaze += alpha * loss_reg_pitch
                    loss_yaw_gaze += alpha * loss_reg_yaw

                    sum_loss_pitch_gaze += loss_pitch_gaze
                    sum_loss_yaw_gaze += loss_yaw_gaze

                    loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                    grad_seq = \
                        [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]

                    optimizer_gaze.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss_seq, grad_seq)
                    optimizer_gaze.step()

                    iter_gaze += 1

                    if (i+1) % 100 == 0:
                        print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                            'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                                epoch+1,
                                num_epochs,
                                i+1,
                                len(dataset)//batch_size,
                                sum_loss_pitch_gaze/iter_gaze,
                                sum_loss_yaw_gaze/iter_gaze
                            )
                            )

                

                # Save models at numbered epochs.
                if epoch % 1 == 0 and epoch < num_epochs:
                    print('Taking snapshot...',
                        torch.save(model.state_dict(),
                                    output+'/fold' + str(fold) +'/'+
                                    '_epoch_' + str(epoch+1) + '.pkl')
                        )

    # args:
    #   arch: default
    #   snapshot: pretrained model
    #


    elif data_set == "socialai":
        min_error_pitch_yaw = 0
        epoch_start_i = 0
        print('you are in dataset of socialai')
        model, pre_url = getArch_weights(args.arch, 90)

        if os.path.exists('checkpoint/latest_model.pth'):
        #if os.path.exists('./checkpoint/latest_model.pth'):
            print('load model from ./checkpoint/latest_model.pth ...')
            checkpoint = torch.load('checkpoint/latest_model.pth')
            #checkpoint = torch.load('./checkpoint/latest_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer_gaze.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start_i = checkpoint['epoch'] + 1
            min_error_pitch_yaw = checkpoint['min_error_pitch_yaw']
            print('Pre-trained model found and recovered!')
        elif args.snapshot == '':
            # load_filtered_state_dict(model, model_zoo.load_url(pre_url))
            print("ERROR, pretrained model required")
            exit()
        else:
            print("loading model from %s ..." % args.snapshot)
            saved_state_dict = torch.load(args.snapshot)
            model.load_state_dict(saved_state_dict)

        # model = nn.DistributedDataParallel(model)
        model.cuda(gpu)
        train_dataset = datasets_local.SocialAI(transform = transformations, train=True, training_val=True, high_res=True)#training_val=True
        print('Loading data.')

        train_loader_gaze = DataLoader(
            dataset=train_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        eval_dataset = datasets_local.SocialAI(transform = transformations, train=True, training_val=False, high_res=True)#training_val=False
        eval_loader_gaze = DataLoader(
            dataset=eval_dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True,)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', int(time.time()))
        output = os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        criterion = nn.CrossEntropyLoss().cuda(gpu)
        reg_criterion = nn.MSELoss().cuda(gpu)
        softmax = nn.Softmax(dim=1).cuda(gpu)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)




        # just train final layers
        # Optimizer gaze
        # optimizer_gaze = torch.optim.Adam([
        #     {'params': get_ignored_params(model), 'lr': 0},
        #     {'params': get_non_ignored_params(model), 'lr': 0}, # args.lr in the original training
        #     {'params': get_fc_params(model), 'lr': args.lr}
        # ], args.lr)
        
   
        # train all layers 
        #    optimizer_gaze = torch.optim.Adam([
        #        {'params': get_ignored_params(model), 'lr': 0},
        #       {'params': get_non_ignored_params(model), 'lr': lr},  # args.lr in the original training ch
        #       {'params': get_fc_params(model), 'lr': lr}
        #   ], lr, weight_decay=1e-4)

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)
        df = pd.DataFrame(columns=['epoch','train_yaw','train_pitch','val_yaw','val_pitch'])
        epoch_train = []
        yaw_error_train = []
        pitch_error_train = []
        epoch_val = []
        yaw_error_val = []
        pitch_error_val = []
        for epoch in range(epoch_start_i, num_epochs):
            model.train()
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
            lr = poly_lr_scheduler(args.lr, iter=epoch, max_iter=args.num_epochs)
            # lr1 = poly_lr_scheduler(args.lr/10, iter=epoch, max_iter=args.num_epochs)
            
        # optimizer_gaze = torch.optim.Adam([
        #     {'params': get_ignored_params(model), 'lr': 0},
        #     {'params': get_non_ignored_params(model), 'lr': 0}, # args.lr in the original training
        #     {'params': get_fc_params(model), 'lr': args.lr}
        # ], args.lr)
            
            optimizer_gaze = torch.optim.Adam([
                 {'params': get_ignored_params(model), 'lr': 0},
                 {'params': get_non_ignored_params(model), 'lr': 0},  # args.lr in the original training ch
                 {'params': get_fc_params(model), 'lr': lr}
            ], lr, weight_decay=1e-4)
            for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(train_loader_gaze):
                # if images_gaze == None:
                #   break
                
                images_gaze = Variable(images_gaze).cuda(gpu)
                # print(images_gaze.shape)
                # Binned labels
                label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
                # print("point one")
                # Continuous labels
                label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                yaw, pitch= model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = \
                    torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                yaw_predicted = \
                    torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                loss_reg_pitch = reg_criterion(
                    pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(
                    yaw_predicted, label_yaw_cont_gaze)
                # print("point two")
                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()
                # scheduler.step()
                # print("point three")
                iter_gaze += 1

                if (i + 1) % 500 == 0:
                    print('Epoch [%d/%d], len dataset:%d , Iter [%d/%d] Losses: '
                          'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                              epoch + 1,
                              num_epochs,
                              len(train_dataset),
                              i + 1,
                              len(train_dataset) // batch_size,
                              sum_loss_yaw_gaze / iter_gaze,
                              sum_loss_pitch_gaze / iter_gaze
                          )
                          )
                if (i + 1) % 2000 == 0:
                    
                    train_yaw=(sum_loss_yaw_gaze / iter_gaze)
                    train_pitch=(sum_loss_pitch_gaze / iter_gaze)
            # # ----------------         validation step          ------------------------
            if epoch % int(validation_step) == 0:# and epoch != 0:
                print("start validation")
                with torch.no_grad():
                    sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0
                    model.eval()
                    for i, (images_gaze, labels_gaze, cont_labels_gaze) in enumerate(eval_loader_gaze):
                        # if images_gaze == None:
                        #     break

                        images_gaze = Variable(images_gaze).cuda(gpu)
                        # print(images_gaze.shape)
                        # Binned labels
                        label_pitch_gaze = Variable(labels_gaze[:, 0]).cuda(gpu)
                        label_yaw_gaze = Variable(labels_gaze[:, 1]).cuda(gpu)
                        # print("point one")
                        # Continuous labels
                        label_pitch_cont_gaze = Variable(cont_labels_gaze[:, 0]).cuda(gpu)
                        label_yaw_cont_gaze = Variable(cont_labels_gaze[:, 1]).cuda(gpu)

                        yaw, pitch = model(images_gaze)

                        # Cross entropy loss
                        loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                        loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                        # MSE loss
                        pitch_predicted = softmax(pitch)
                        yaw_predicted = softmax(yaw)

                        pitch_predicted = \
                            torch.sum(pitch_predicted * idx_tensor, 1) * 4 - 180
                        yaw_predicted = \
                            torch.sum(yaw_predicted * idx_tensor, 1) * 4 - 180

                        loss_reg_pitch = reg_criterion(
                            pitch_predicted, label_pitch_cont_gaze)
                        loss_reg_yaw = reg_criterion(
                            yaw_predicted, label_yaw_cont_gaze)
                        # print("point two")
                        # Total loss
                        loss_pitch_gaze += alpha * loss_reg_pitch
                        loss_yaw_gaze += alpha * loss_reg_yaw

                        sum_loss_pitch_gaze += loss_pitch_gaze
                        sum_loss_yaw_gaze += loss_yaw_gaze

                        loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                        # grad_seq = [torch.tensor(1.0).cuda(gpu) for _ in range(len(loss_seq))]
                        # scheduler.step()
                        # print("point three")
                        iter_gaze += 1

                    print('Validation: Batch size: %d, Test lenght %d, Losses: '
                          'Gaze Yaw %.4f,Gaze Pitch %.4f' % (
                              16,
                              len(eval_dataset),
                              sum_loss_yaw_gaze / iter_gaze,
                              sum_loss_pitch_gaze / iter_gaze
                          )
                          )
                    
                    val_yaw = sum_loss_yaw_gaze / iter_gaze
                    val_pitch = sum_loss_pitch_gaze / iter_gaze

                    print('Taking snapshot...',
                          torch.save(model.state_dict(),
                                     output + '/' +
                                     '_epoch_' + str(epoch + 1) + '.pkl')
                          )
                    total_error = sum_loss_pitch_gaze / iter_gaze + sum_loss_yaw_gaze / iter_gaze
                    if min_error_pitch_yaw == 0:
                        min_error_pitch_yaw = total_error
                    elif total_error < min_error_pitch_yaw:
                        min_error_pitch_yaw = total_error
                        torch.save(model.state_dict(),
                                   os.path.join(output, 'best_model.pth'))
                        print("Found a better model at epoch", epoch + 1,". Best model updated --> min_avg_error: ", min_error_pitch_yaw)


            # save checkpoint
            # row_data = {'epoch':epoch,'train_yaw':train_yaw,'train_pitch':train_pitch,'val_yaw':val_yaw,'val_pitch':val_pitch}
            # df = df.append(row_data, ignore_index=True)
            # df.to_csv('resultlin.csv',index=False)
			
            if epoch % 1 == 0:# and epoch != 0:
                if not os.path.isdir("checkpoint"):
                    os.mkdir("checkpoint")

                state = {
                    "epoch": epoch,
                    "min_error_pitch_yaw": min_error_pitch_yaw,
                    "model_state_dict": model.state_dict(),
                    'optimizer_state_dict': optimizer_gaze.state_dict(),
                }
                # print(state)
                torch.save(state,
                           os.path.join("checkpoint", 'latest_model.pth'))
                print('Checkpoint saved')

        os.remove(os.path.join("checkpoint", 'latest_model.pth'))
        #df['epoch_train'] = epoch_train
        #df['yaw_error_train'] = yaw_error_train
        #df['pitch_error_train'] = pitch_error_train
        #df['epoch_val'] = epoch_val
        #df['yaw_error_val'] = yaw_error_val
        #df['pitch_error_val'] = pitch_error_val
        #df.to_csv('epoch_result_new.csv')
        print(epoch_train,yaw_error_train,pitch_error_train,epoch_val,yaw_error_val,pitch_error_val)

