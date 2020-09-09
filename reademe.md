We use tensorflow 1.5 with python 3.5.
You can refer to setting_cuda9_cudnn7_tensorflow1.5.sh to build up your environment. 

synthetic_toy.py

    This code can be used to reproduce the results in Figure 1. 
     
        python3 synthetic_toy.py -bUseSND=True  -bLipReg=False -bMaxGrad=False               #for SN        (shown in figure)
        python3 synthetic_toy.py -bUseSND=False -bLipReg=True  -bMaxGrad=False -sGP_type=gp  #for GP        (shown in figure)
        python3 synthetic_toy.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  #for MAXGP     (shown in figure) 
        python3 synthetic_toy.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=al  #for MAXAL    

synthetic_real.py:

    This code can be used to reproduce the results in Figure 2.
    
        python3 synthetic_real.py -bUseSND=True  -bLipReg=False -bMaxGrad=False               #for SN
        python3 synthetic_real.py -bUseSND=False -bLipReg=True  -bMaxGrad=False -sGP_type=gp  #for GP       (shown in figure) 
        python3 synthetic_real.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  #for MAXGP    (shown in figure)
        python3 synthetic_real.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=al  #for MAXAL

realdata_resnet.py:
    
    This code can be used to reproduce the results in Figure 3 and Figure4.
    
    Use "-sGAN_type=[x, log_sigmoid, hinge]" to switch different objective, e.g.,:

        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=0.1  -sGAN_type=x              # for wgan loss with maxgp
        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=0.1  -sGAN_type=log_sigmoid    # for vanilla gan loss with maxgp
        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=0.1  -sGAN_type=hinge          # for hinge loss with maxgp

    Use "-fWeightLip=[0.1, 1.0, 10.0]" to set different weight of regularization $\rho$, e.g.,:
    
        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=0.1   -sGAN_type=x           
        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=1.0   -sGAN_type=x          
        python3 realdata_resnet.py -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  -fWeightLip=10.0  -sGAN_type=x 
       
    Use "-bUseSND=True  -bLipReg=False -bMaxGrad=False" etc., to swtich to different type of regularization, i.e.,
    
        -bUseSND=True  -bLipReg=False -bMaxGrad=False  # for SN
        -bUseSND=False -bLipReg=True  -bMaxGrad=False -sGP_type=gp  #for GP 
        -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=gp  #for MAXGP
        -bUseSND=False -bLipReg=True  -bMaxGrad=True  -sGP_type=al  #for MAXAL         
          
        
To try MaxGP with buffered, add the following flag: 
    
    -fBufferBatch=0.25
    or
    -fBufferBatch=-0.25
 
    The buffer size equals to fBufferBatch * iBatchSize; 
    The sign of fBufferBatch indicates the way we use the buffer: 
        postive -> extend the batch: batch size for maxgp becomes iBatchSize * (1+fBufferBatch) 
        negative -> insert into the batch: keep the batchsize of gp unchanged.  
        
Corresponding results will be in the ../result folder. 