import torch
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from math import inf

def error_criterion(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    train_error = (max_indices != labels).sum().data[0]/max_indices.size()[0]
    return train_error

def evalaute_mdl_data_set(loss,error,net,dataloader,enable_cuda,iterations=inf):
    '''
    Evaluate the error of the model under some loss and error with a specific data set.
    '''
    running_loss,running_error = 0,0
    for i,data in enumerate(dataloader):
        if i >= iterations:
            break
        inputs, labels = extract_data(enable_cuda,data,wrap_in_variable=True)
        outputs = net(inputs)
        running_loss += loss(outputs,labels).data[0]
        running_error += error(outputs,labels)
    return running_loss/(i+1),running_error/(i+1)

def extract_data(enable_cuda,data,wrap_in_variable=False):
    inputs, labels = data
    if enable_cuda:
        inputs, labels = inputs.cuda(), labels.cuda() #TODO potential speed up?
    if wrap_in_variable:
        inputs, labels = Variable(inputs), Variable(labels)
    return inputs, labels

def train_and_track_stats(nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion, iterations=inf):
    enable_cuda = args.enable_cuda
    ''' Add stats before training '''
    train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, trainloader, enable_cuda, iterations)
    test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, testloader, enable_cuda, iterations)
    print(f'[-1, -1], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
    ##
    ''' Start training '''
    print('about to start training')
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        running_train_loss,running_train_error = 0.0,0.0
        M_train = 0
        for i,data_train in enumerate(trainloader):
            ''' zero the parameter gradients '''
            optimizer.zero_grad()
            ''' train step = forward + backward + optimize '''
            inputs, labels = extract_data(enable_cuda,data_train,wrap_in_variable=True)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.data[0]
            running_train_error += error_criterion(outputs,labels)
        ''' End of Epoch: collect stats'''
        train_loss_epoch, train_error_epoch = running_train_loss/(i+1), running_train_error/(i+1)
        #train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion,error_criterion,net,trainloader,enable_cuda,iterations)
        test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion,error_criterion,net,testloader,enable_cuda,iterations)
        print(f'[{epoch}, {i+1}], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
    return train_loss_epoch, train_error_epoch, test_loss_epoch, test_error_epoch

def main():
    ##
    ''' Get Data set '''
    data_path = './data'
    transform = [transforms.ToTensor(),transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )]
    transform = transforms.Compose(transform)
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=shuffle_train, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=suffle_test, num_workers=num_workers)
    ''' Get model '''
    C,H,W = 3,32,32
    net = nn.Sequential(
        torch.nn.Conv2d(C,13,5), #(in_channels, out_channels, kernel_size),
        torch.nn.Linear(13, 10)
    )
    ''' Train '''
    nb_epochs = 10
    lr = 0.1
    error_criterion = error_criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.0)
    train_and_track_stats(nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion, iterations=inf)