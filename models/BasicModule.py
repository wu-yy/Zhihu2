#coding:utf8

import torch as t
import time

#class BasicModule(t.nn.Module):

class BasicModule(t.nn.Module):

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))

    def load(self,path,change_opt=True):
        print(path)
        data=t.load(path)
        if 'opt' in data:

            if change_opt:
                self.opt.parse(data['opt'],print_=False)
                self.opt.embedding_path=None
                self.__init__(self.opt)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)

        return self
    def save(self,name=None,new=False):
        prefix='checkpoints/'+self.model_name+"_"+self.opt.type_+"_"

        if name is None:
            name=time.strftime('%m%d_%H:%M:%S.pth')
        path=prefix+name

        if new:
            data={'opt':self.opt.state_dict(),'d':self.state_dict()}
        else:
            data=self.state_dict()

        t.save(data,path)
        return path

    def get_optimizer(self,lr1,lr2=0,weight_decay=0):
        ignored_params=list(map(id,self.encoder.parameters()))

        base_params=filter(lambda p :id(p) not in ignored_params,self.parameters())

        if lr2 is None:
            lr2=lr1*0.5

        optimizer=t.optim.Adam([
            dict(params=base_params,weight_decay=weight_decay,lr=lr1),
            {'params':self.encoder.parameters(),'lr':lr2}
        ])

        return optimizer



