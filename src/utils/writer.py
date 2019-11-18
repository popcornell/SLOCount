#  Developed by Samuele Cornell on 7/13/19, 8:10 PM.
#  Last modified 7/11/19, 9:42 PM.
#  Copyright (c) 2019. All rights reserved.

from tensorboardX import SummaryWriter


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_step_training(self, train_loss, step):
        self.add_scalar('/loss/step_train_loss', train_loss, step)

    def log_step_val(self, val_loss, step):

        self.add_scalar('/loss/step_val_loss', val_loss, step)


    def log_metrics(self, name, loss, fa, miss, recall, precision, f1, matt, tp, tn, fp, fn, step):
        temp = "loss/{}_loss".format(name) # workaround otherwise tensorboard not works
        self.add_scalar(temp, loss, step)
        self.add_scalar('fa/{}_fa'.format(name), fa, step)
        self.add_scalar('miss/{}_miss'.format(name), miss, step)
        #self.add_scalar('acc/{}_acc'.format(name), acc, step)
        #self.add_scalar('der/{}_der'.format(name), der, step)
        self.add_scalar('f1/{}_f1'.format(name), f1, step)



        self.add_scalar('precision/{}_precision'.format(name), precision, step)
        self.add_scalar('recall/{}_recall'.format(name), recall, step)
        self.add_scalar('matt/{}_matt'.format(name), matt, step)
        self.add_scalar('tp/{}_tp'.format(name), tp, step)
        self.add_scalar('tn/{}_tn'.format(name), tn, step)
        self.add_scalar('fp/{}_matt'.format(name), fp, step)
        self.add_scalar('fn/{}_matt'.format(name), fn, step)









