import torch
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from math import inf

from pdb import set_trace as st

def error_criterion(outputs,labels):
    max_vals, max_indices = torch.max(outputs,1)
    error = (max_indices != labels).float().sum()/max_indices.size()[0]
    return error.data[0]

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

def train_and_track_stats(enable_cuda, nb_epochs, trainloader,testloader, net,optimizer,criterion,error_criterion, iterations=inf):
    ''' Add stats before training '''
    train_loss_epoch, train_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, trainloader, enable_cuda, iterations)
    test_loss_epoch, test_error_epoch = evalaute_mdl_data_set(criterion, error_criterion, net, testloader, enable_cuda, iterations)
    print(f'[-1, -1], (train_loss: {train_loss_epoch}, train error: {train_error_epoch}) , (test loss: {test_loss_epoch}, test error: {test_error_epoch})')
    ##
    ''' Start training '''
    print('about to start training')
    for epoch in range(nb_epochs):  # loop over the dataset multiple times
        running_train_loss,running_train_error = 0.0,0.0
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

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def main():
    enable_cuda = True
    print('running main')
    num_workers = 0
    ''' Get Data set '''
    batch_size_test = 10000
    batch_size_train = 10000
    data_path = './data'
    transform = [transforms.ToTensor(),transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )]
    transform = transforms.Compose(transform)
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,shuffle=True, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=False, num_workers=num_workers)
    ''' Get model '''
    net = torch.nn.Sequential(
        torch.nn.Conv2d(3,13,5), #(in_channels, out_channels, kernel_size),
        Flatten(),
        torch.nn.Linear(28*28*13, 13),
        torch.nn.Linear(13, 10)
    )
    net.cuda()
    ''' Train '''
    nb_epochs = 10
    lr = 0.1
    err_criterion = error_criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.0)
    train_and_track_stats(enable_cuda, nb_epochs, trainloader,testloader, net,optimizer,criterion,err_criterion, iterations=inf)
    ''' Done '''
    print('Done')

if __name__ == '__main__':
    main()