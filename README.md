<<<<<<< HEAD
# Code for the ACM/IEEE TinyML Contest at ICCAD 2022

## File Structure

> + Folder for model training design: ./Training
> + Folder for C source code (including the whole project): ./Deployment
> + Trained model weights in tflite: ./Weight/best_weight.tflite

## How to run?
### Training

Set up environment
    
    pip install tensorflow
    
and running
    
    cd Training
    python select_model.py


### Deploy

For CubeMX  
```
1.Import best_weight.tflite, use STM32RunTIME and low level compression  
2.In advanced setting, click on ONLY "Use activation buffer for input buffer" & "Use activation buffer for the output buffer" 
```
For Keil Compiling Settings
```
1.Option for Target->Target->Code generation：Use default compiler version 6 & Use Micro LIB;
2.Option for Target->C/C++(AC6)->Optimization:-Oz
3.Option for Target->C/C++(AC6)->Click on "One ELF Section per Function" & "Link Time Optimization" & "Execute-only Code" & "Short enums/wchar"
```


=======
# TinyML_Boolean
>>>>>>> ee60a040edc004fd86de75e4fae470d0fc800ee1

1. Researching on developing more sophisticated pruning and quantization technique for characterization of biases for the
compressed model.

2. Implemented a Lottery Ticket Hypothesis-based model pruning on the ECG data, deployed and obtained inference on
STM32F303K8T6 (ARM Cortex M4 core).
