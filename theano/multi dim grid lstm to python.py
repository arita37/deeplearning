# -*- coding: utf-8 -*-
"""
Torch to Theano Python

grid-lstm-tensorflow
Examples of using GridLSTM (and GridRNN in general) in tensorflow

The GridRNN implementation in tensorflow is generic, in the sense that it supports multiple dimensions with various settings for input/output dimensions, priority dimensions and non-recurrent dimensions. The type of recurrent cell can also be selected among LSTM, GRU or vanilla RNN.

Here we collect some examples that demonstrate GridRNN, which will be added over time. The current list of examples include:

char-rnn: 2GridLSTM for character-level language modeling.

"""


########################### grid-lstm/model/GridLSTM.lua    #########################
import 'nn'
import 'nngraph'
'''
  This is called once per dimension inside a grid LSTM block to create the gated
  update of the dimension's hidden state and memory cell.
  It takes h_t and h_d, the hidden states from the temporal and 
  depth dimensions respectively, as well as prev_c, the  dimension's previous memory cell.
  
  It returns next_c, next_h along the dimension, using a standard lstm gated update, 
  conditioned on the concatenated time and  depth hidden states.
'''
def lstm(h_t, h_d, prev_c, rnn_size)
  all_input_sums = nn.CAddTable()({h_t, h_d})
  reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  
  # decode the gates
  in_gate = nn.Sigmoid()(n1)
  forget_gate = nn.Sigmoid()(n2)
  out_gate = nn.Sigmoid()(n3)

  # decode the write inputs
  in_transform = nn.Tanh()(n4)

  # perform the LSTM update
  next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })

  # gated cells form the output
  next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h



'''
  GridLSTM:
    1) Map input x into memory and hidden cells m(1), h(1) along the depth dimension.
    2) Concatenate previous hidden states from time and depth dimensions, [h(1), h(2)] into H.
    3) Forward the time LSTM, LSTM_2(H) -> h(2)', m(2)'.
    4) Concatenate transformed h(2)' and h(1) into H' = [h(1), h(2)']
    5) Forward the depth LSTM, LSTM_1(H') -> h(1)', m(1)'
    6) Either repeat 2-5 for another layer or map h(1)', the final hidden state along the depth 
       dimension, to a character prediction.
  '''
GridLSTM = {}
def GridLSTM.grid_lstm(input_size, rnn_size, n, dropout, should_tie_weights)
  dropout = dropout or 0 

  # There will be 2*n+1 inputs
  inputs = {}
  table.insert(inputs, nn.Identity()()) # input c for depth dimension
  table.insert(inputs, nn.Identity()()) # input h for depth dimension
  for L = 1,n  : 
    table.insert(inputs, nn.Identity()()) # prev_c[L] for time dimension
    table.insert(inputs, nn.Identity()()) # prev_h[L] for time dimension
  

  shared_weights
  if should_tie_weights == 1  :  
      shared_weights = {nn.Linear(rnn_size, 4 * rnn_size), nn.Linear(rnn_size, 4 * rnn_size)} 

  outputs_t = {} # Outputs being handed to the next time step along the time dimension
  outputs_d = {} # Outputs being handed from one layer to the next along the depth dimension

  for L in range(1,n)  : 
    # Take hidden and memory cell from previous time steps
    prev_c_t = inputs[L*2+1]
    prev_h_t = inputs[L*2+2]

    if L == 1  : 
      # We're in the first layer
      prev_c_d = inputs[1] # input_c_d: the starting depth dimension memory cell, just a zero vec.
      prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) 
      # input_h_d: the starting depth dim hidden state. map a char into hidden space via a lookup table
    else 
      # We're in the higher layers 2...N
      # Take hidden and memory cell from layers below
      prev_c_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0  :  prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L}  # apply dropout, if any
    

    # Evaluate the input sums at once for efficiency
    t2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}
    d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    
    # Get transformed memory and hidden states pointing in the time direction first
    next_c_t, next_h_t = lstm(t2h_t, d2h_t, prev_c_t, rnn_size)

    # Pass memory cell and hidden state to next timestep
    table.insert(outputs_t, next_c_t)
    table.insert(outputs_t, next_h_t)

    # Evaluate the input sums at once for efficiency
    t2h_d = nn.Linear(rnn_size, 4 * rnn_size)(next_h_t):annotate{name='i2h_'..L}
    d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}

    # See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
    # The weights along the temporal dimension are already tied (cloned many times in train.lua)
    # Here we can tie the weights along the depth dimension. Having invariance in computation
    # along the depth appears to be critical to solving the 15 digit addition problem w/ high accy.
    # See fig 4. to compare tied vs untied grid lstms on this task.
    if should_tie_weights == 1  : 
      print("tying weights along the depth dimension")
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    
    
    # Create the lstm gated update pointing in the depth direction.
    # We 'prioritize' the depth dimension by using the updated temporal hidden state as input
    # instead of the previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
    next_c_d, next_h_d = lstm(t2h_d, d2h_d, prev_c_d, rnn_size)

    # Pass the depth dimension memory cell and hidden state to layer above
    table.insert(outputs_d, next_c_d)
    table.insert(outputs_d, next_h_d)
  

  # set up the decoder
  top_h = outputs_d[#outputs_d]
  if dropout > 0  :  top_h = nn.Dropout(dropout)(top_h) 
  proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs_t, logsoft)

  return nn.gModule(inputs, outputs_t)


return GridLSTM






########################### grid-lstm/model/GridLSTM.lua    #########################
######## grid-lstm/train.lua
''' 
This file trains a character-level multi-layer RNN on text data
Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)
''' 

'torch'
import 'nn'
import 'nngraph'
import 'optim'
import 'lfs'
import 'cudnn'

import 'util.OneHot'
import 'util.misc'
CharSplitLMMinibatchLoader = import 'util.CharSplitLMMinibatchLoader'
model_utils = import 'util.model_utils'
LSTM = import 'model.LSTM'
GridLSTM = import 'model.GridLSTM'
GRU = import 'model.GRU'
RNN = import 'model.RNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
# data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
# task params
cmd:option('-task', 'char', 'task to train on: char, addition')
cmd:option('-digit_length', 4, 'length of the digits to add for the addition task')
# model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, grid_lstm, gru, or rnn')
cmd:option('-tie_weights', 1, 'tie grid lstm weights?')
# optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            # test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
# bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing',0,'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
# GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

# parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
# train / val / test split for data, in fractions
test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

# initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0  : 
    ok, cunn = pcall(require, 'cunn')
    ok2, cutorch = pcall(require, 'cutorch')
    if not ok  :  print('package cunn not found!') 
    if not ok2  :  print('package cutorch not found!') 
    if ok and ok2  : 
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting
    


# initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1  : 
    ok, cunn = pcall(require, 'clnn')
    ok2, cutorch = pcall(require, 'cltorch')
    if not ok  :  print('package clnn not found!') 
    if not ok2  :  print('package cltorch not found!') 
    if ok and ok2  : 
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting
    


# create the data loader class
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
vocab_size = loader.vocab_size  # the number of distinct characters
vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
# make sure output directory exists
if not path.exists(opt.checkpoint_dir)  :  lfs.mkdir(opt.checkpoint_dir) 

# define the model: prototypes for one timestep,  :  clone them in time
do_random_init = true
if string.len(opt.init_from) > 0  : 
    print('loading a model from checkpoint ' .. opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    # make sure the vocabs are the same
    vocab_compatible = true
    checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i)  : 
            vocab_compatible = false
        
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    
    if not (checkpoint_vocab_size == vocab_size)  : 
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    
    assert(vocab_compatible, 'error, char voca for this dataset and  one in saved checkpoint are not the same. ')
    # overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm'  : 
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'grid_lstm'  : 
        protos.rnn = GridLSTM.grid_lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout, opt.tie_weights)
        
    elif opt.model == 'gru'  : 
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'rnn'  : 
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    
    protos.criterion = nn.ClassNLLCriterion()




# the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0  :  h_init = h_init:cuda() 
    if opt.gpuid >=0 and opt.opencl == 1  :  h_init = h_init:cl() 
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' or opt.model == 'grid_lstm'  : 
        table.insert(init_state, h_init:clone()) # extra initial state for prev_c
    


# ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0  :   for k,v in pairs(protos) do v:cuda() 

if opt.gpuid >= 0 and opt.opencl == 1  :   for k,v in pairs(protos) do v:cl() 


# put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

# initialization
if do_random_init  : 
    params:uniform(-0.08, 0.08) # small uniform numbers

# initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' or opt.model == 'grid_lstm'  : 
    for layer_idx = 1, opt.num_layers do
        for _,node in ipairs(protos.rnn.forwardnodes) do
            if node.data.annotations.name == "i2h_" .. layer_idx  : 
                print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
                
                # the gates are, in order, i,f,o,g, so f is the 2nd block of weights
                node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
            
        
    
print('number of parameters in the model: ' .. params:nElement())
# make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)


# preprocessing helper function
function prepro(x,y)
    x = x:transpose(1,2):contiguous() # swap the axes for faster indexing
    y = y:transpose(1,2):contiguous()
    if opt.gpuid >= 0 and opt.opencl == 0  :  # ship the input arrays to GPU
        # have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    
    if opt.gpuid >= 0 and opt.opencl == 1  :  # ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    
    return x,y


function get_input_mem_cell()
    input_mem_cell = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >= 0 and opt.opencl == 0  : 
      input_mem_cell = input_mem_cell:float():cuda()
    
    return input_mem_cell


function get_zeroed_d_output_t(vocab_size)
    zeroed_d_output_t = torch.zeros(opt.batch_size, vocab_size)
    if opt.gpuid >= 0 and opt.opencl == 0  : 
      zeroed_d_output_t = zeroed_d_output_t:float():cuda()
    
    return zeroed_d_output_t


# evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    n = loader.split_sizes[split_index]
    if max_batches ~= nil  :  n = math.min(max_batches, n) 

    loader:reset_batch_pointer(split_index) # move batch iteration pointer for this split to front
    loss = 0
    accy = 0
    normal = 0
    rnn_state = {[0] = init_state}

    for i = 1,n do # iterate over batches in the split
        # fetch a batch
        x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        
        
        # forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() # for dropout proper functioning
            if opt.model == "grid_lstm"  : 
              input_mem_cell = get_input_mem_cell()
              rnn_inputs = {input_mem_cell, x[t], unpack(rnn_state[t-1])} 
              # if we're using a grid lstm, hand in a zero vec for the starting memory cell state
            else
              rnn_inputs = {x[t], unpack(rnn_state[t-1])}
            
            lst = clones.rnn[t]:forward(rnn_inputs)
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) 
            prediction = lst[#lst]

            target_delimiter_position = opt.seq_length - (opt.digit_length + 2)
            if opt.task == "addition" and t > target_delimiter_position  : 
                max, pred_argmax = torch.max(prediction,2)
                accy = accy + torch.eq(pred_argmax, y[t]):sum()
                normal = normal + prediction:size(1)
            

            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        
        # carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    

    out
    if opt.task == "addition"  : 
        out = accy / normal
    else 
        out = loss / opt.seq_length / n
    

    return out


# do fwd/bwd and return loss, grad_params
init_state_global = clone_list(init_state)
def feval(x) :
    if x ~= params  : 
        params:copy(x)
    
    grad_params:zero()

    #----------------# get minibatch -------------------
    x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    
    #----------------- forward pass -------------------
    rnn_state = {[0] = init_state_global}
    predictions = {}           # softmax outputs
    loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() # make sure we are in correct mode (this is cheap, sets flag)
        rnn_inputs
        if opt.model == "grid_lstm"  : 
          input_mem_cell = get_input_mem_cell()
          rnn_inputs = {input_mem_cell, x[t], unpack(rnn_state[t-1])} # if we're using a grid lstm, hand in a zero vec for the starting memory cell state
        else
          rnn_inputs = {x[t], unpack(rnn_state[t-1])}
        
        lst = clones.rnn[t]:forward(rnn_inputs)
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i])  # extract the state, without output
        predictions[t] = lst[#lst] # last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    
    loss = loss / opt.seq_length



    ----------------# backward pass -------------------
    # initialize gradient at time t to be zeros (there's no influence from future)
    drnn_state = {[opt.seq_length] = clone_list(init_state, true)} # true also zeros the clones
    for t=opt.seq_length,1,-1 do

        # If we do addition task and we're at t < position of target delimiter, just use a vec of zeros for dL/dOutput
        # We don't want to suffer prediction loss prior to the target delimiter, just recurrence loss.
        target_delimiter_position = opt.seq_length - (opt.digit_length + 2)
        if opt.task == "addition" and t < target_delimiter_position  : 
            doutput_t = get_zeroed_d_output_t(loader.vocab_size)
        else
            doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        

        # backprop through loss, and softmax/linear
        table.insert(drnn_state[t], doutput_t) # drnn_state[t] already has dL/dH_t+1 vectors for every layer; just adding the dL/dOutput to the list. 

        dlst = clones.rnn[t]:backward(rnn_inputs, drnn_state[t]) # <- right here, you're apping the doutput_t to the list of dLdh for all layers,  :  using that big list to backprop into the input and unpacked rnn state vecs at t-1
        drnn_state[t-1] = {}
        skip_index
        if opt.model == "grid_lstm"  :  skip_index = 2 else skip_index = 1 
        for k,v in pairs(dlst) do
            if k > skip_index  :  # k <= skip_index is gradient on inputs, which we dont need
                # note we do k-1 because first item is dembeddings, and  :  follow the 
                # derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-skip_index] = v
            
        
    
    ----------------------# misc ----------------------
    # transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] # NOTE: I don't think this needs to be a clone, right?
    # grad_params:div(opt.seq_length) # this line should be here but since we use rmsprop it would have no effect. Removing for efficiency
    # clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params


# start optimization here
train_losses = {}
val_losses = {}
optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
iterations = opt.max_epochs * loader.ntrain
iterations_per_epoch = loader.ntrain
loss0 = nil
for i = 1, iterations do
    epoch = i / loader.ntrain

    timer = torch.Timer()
    _, loss = optim.rmsprop(feval, params, optim_state)
    if opt.accurate_gpu_timing == 1 and opt.gpuid >= 0  : 
        ''' 
        Note on timing: The reported time can be off because the GPU is invoked async. If one
        wants to have exactly accurate timings one must call cutorch.synchronize() right here.
        I will avoid doing so by default because this can incur computational overhead.
        ''' 
        cutorch.synchronize()
    
    time = timer:time().real
    
    train_loss = loss[1] # the loss is inside a list, pop it
    train_losses[i] = train_loss

    # exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1  : 
        if epoch >= opt.learning_rate_decay_after  : 
            decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor # decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        
    

    # every now and  :  or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations  : 
        # evaluate loss on validation data
        val_loss = eval_split(2) # 2 = validation
        val_losses[i] = val_loss

        savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    

    if i % opt.print_every == 0  : 
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    
   
    if i % 10 == 0  :  collectgarbage() 

    # handle early stopping if things are going really bad
    if loss[1] ~= loss[1]  : 
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break # halt
    
    if loss0 == nil  :  loss0 = loss[1] 
    if loss[1] > loss0 * 3  : 
        print('loss is exploding, aborting.')
        break # halt
    


!pip install chainer

!pip install -U chainer -vvvv


!set VS100COMNTOOLS=%VS120COMNTOOLS%`

!set PATH= %VS120COMNTOOLS%\..\..\VC\bin;%PATH%` 






##################################################################
##########  grid-lstm/sample.lua
''' 
This file samples characters from a trained model
Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
''' 

import 'torch'
import 'nn'
import 'nngraph'
import 'optim'
import 'lfs'

import 'util.OneHot'
import 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
# required:
cmd:argument('-model','model checkpoint to use for sampling')
# optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used asprompt to "seed" state the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

# parse input params
opt = cmd:parse(arg)

# gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1  :  print(str) 


# check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0  : 
    ok, cunn = pcall(require, 'cunn')
    ok2, cutorch = pcall(require, 'cutorch')
    if not ok  :  gprint('package cunn not found!') 
    if not ok2  :  gprint('package cutorch not found!') 
    if ok and ok2  : 
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure saved checkpoint wastrained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting
    


# check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1  : 
    ok, cunn = pcall(require, 'clnn')
    ok2, cutorch = pcall(require, 'cltorch')
    if not ok  :  print('package clnn not found!') 
    if not ok2  :  print('package cltorch not found!') 
    if ok and ok2  : 
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting
    


torch.manualSeed(opt.seed)

# load the model checkpoint
if not lfs.attributes(opt.model, 'mode')  : 
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prep cv/ ?')

checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() # put in eval mode so that dropout works properly

# initialize the vocabulary (and its inverted version)
vocab = checkpoint.vocab
ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c 

# initialize the rnn state to all zeros
gprint('creating an ' .. checkpoint.opt.model .. '...')
current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    # c and h for all layers
    h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    if opt.gpuid >= 0 and opt.opencl == 0  :  h_init = h_init:cuda() 
    if opt.gpuid >= 0 and opt.opencl == 1  :  h_init = h_init:cl() 
    table.insert(current_state, h_init:clone())
    if checkpoint.opt.model == 'grid_lstm'  : 
        table.insert(current_state, h_init:clone()) # extra initial state for prev_c
    

state_size = #current_state

function get_input_mem_cell()
    input_mem_cell = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 and opt.opencl == 0  : 
      input_mem_cell = input_mem_cell:float():cuda()
    
    return input_mem_cell


# do a few seeded timesteps
seed_text = opt.primetext
if string.len(seed_text) > 0  : 
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]])
        if opt.gpuid >= 0 and opt.opencl == 0  :  prev_char = prev_char:cuda() 
        if opt.gpuid >= 0 and opt.opencl == 1  :  prev_char = prev_char:cl() 
        if checkpoint.opt.model == "grid_lstm"  : 
          input_mem_cell = get_input_mem_cell()
          rnn_inputs = {input_mem_cell, prev_char, unpack(current_state)} # if we're using a grid lstm, hand in a zero vec for the starting memory cell state
        else
          rnn_inputs = {prev_char, unpack(current_state)}
        
        lst = protos.rnn:forward(rnn_inputs)
        # lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) 
        prediction = lst[#lst] # last element holds the log probabilities
    
else
    # fill with uniform probabilities over characters (? hmm)
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
    if opt.gpuid >= 0 and opt.opencl == 0  :  prediction = prediction:cuda() 
    if opt.gpuid >= 0 and opt.opencl == 1  :  prediction = prediction:cl() 


# start sampling/argmaxing
for i=1, opt.length do

    # log probabilities from the previous timestep
    if opt.sample == 0  : 
        # use argmax
        _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        # use sampling
        prediction:div(opt.temperature) # scale by temperature
        probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) # renormalize so probs sum to one
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    

    # forward the rnn for next character
    if checkpoint.opt.model == "grid_lstm"  : 
      input_mem_cell = get_input_mem_cell()
      rnn_inputs = {input_mem_cell, prev_char, unpack(current_state)} # if we're using a grid lstm, hand in a zero vec for the starting memory cell state
    else
      rnn_inputs = {prev_char, unpack(current_state)}
    
    lst = protos.rnn:forward(rnn_inputs)
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) 
    prediction = lst[#lst] # last element holds the log probabilities

    io.write(ivocab[prev_char[1]])

io.write('\n') io.flush()






















################################################################################
##############Tensor Flow Model ###############################################
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/grid_rnn/python/kernel_tests/grid_rnn_test.py


# model.py

import numpy as np, tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq
from tensorflow.contrib import grid_rnn


class Model(object):
  '''
  batchsize,  seq_length,  model "gridlstm",  
  rnn_size, vocab_size,  num_layers, 
  '''    
    
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        additional_cell_args = {}
        if args.model == 'rnn':          cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':        cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':       cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'gridlstm':  
                    cell_fn = grid_rnn.Grid2LSTMCell
                    additional_cell_args.update({'use_peepholes': True, 'forget_bias': 1.0})
        elif args.model == 'gridgru':    cell_fn = grid_rnn.Grid2GRUCell
        else:       raise Exception("model type not supported: {}".format(args.model))



        cell = cell_fn(args.rnn_size, **additional_cell_args)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)



        with tf.variable_scope('rnnlm'):
            #input softmax:  SentenceSize x VocaSize
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            
            #Ouput  1 word/character ahead 
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

            #Format the input
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        #Loop over the sequence
        def loop(prev, _):
            prev = tf.nn.xw_plus_b(prev, softmax_w, softmax_b)
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)


        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                                  loop_function=loop if infer else None, scope='rnnlm')

        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        
        
        self.logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)


        #Loss function and training by gradient diffusion
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.ones([args.batch_size * args.seq_length])],
                                                args.vocab_size)
                                                
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length  #Sum_Loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))



    #Generate Sample
    def sample(self, sess, chars, vocab, num=200, prime='The '):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return (int(np.searchsorted(t, np.random.rand(1) * s)))

        ret = prime
        char = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            # sample = int(np.random.choice(len(p), p=p))
            sample = weighted_pick(p)
            pred = chars[sample]
            ret += pred
            char = pred
        return ret
        
        
        
        
        
        
        
        
        
        
        
####################################################################################       
####################################################################################   
# train.py
        
import argparse, cPickle, os, time
import tensorflow as tf, pandas as pd
from model import Model
from utils import TextLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, gridlstm, gridgru')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    args = parser.parse_args()
    train(args)


def train(args):
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'w') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    model = Model(args)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        train_loss_iterations = {'iteration': [], 'epoch': [], 'train_loss': [], 'val_loss': []}

        for e in xrange(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = model.initial_state.eval()
            for b in xrange(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state}
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                 = time.time()
                batch_idx = e * data_loader.num_batches + b
                print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(batch_idx,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss,  - start)
                train_loss_iterations['iteration'].app(batch_idx)
                train_loss_iterations['epoch'].app(e)
                train_loss_iterations['train_loss'].app(train_loss)

                if batch_idx % args.save_every == 0:

                    # evaluate
                    state_val = model.initial_state.eval()
                    avg_val_loss = 0
                    for x_val, y_val in data_loader.val_batches:
                        feed_val = {model.input_data: x_val, model.targets: y_val, model.initial_state: state_val}
                        val_loss, state_val, _ = sess.run([model.cost, model.final_state, model.train_op], feed_val)
                        avg_val_loss += val_loss / len(data_loader.val_batches)
                    print 'val_loss: {:.3f}'.format(avg_val_loss)
                    train_loss_iterations['val_loss'].app(avg_val_loss)

                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print "model saved to {}".format(checkpoint_path)
                else:
                    train_loss_iterations['val_loss'].app(None)

            pd.DataFrame(data=train_loss_iterations,
                         columns=train_loss_iterations.keys()).to_csv(os.path.join(args.save_dir, 'log.csv'))

if __name__ == '__main__':
    main()









####################################################################################       
####################################################################################   
# utils.py

import cPickle, collections, os, codecs, numpy as np

class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print "reading text file"
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print "loading preprocessed files"
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()


    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = list(zip(*count_pairs))
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'w') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(map(self.vocab.get, data))
        np.save(tensor_file, self.tensor)


    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)



    def create_batches(self):
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

        validation_batches = int(self.num_batches * .2)
        self.val_batches = zip(self.x_batches[-validation_batches:], self.y_batches[-validation_batches:])
        self.x_batches = self.x_batches[:-validation_batches]
        self.y_batches = self.y_batches[:-validation_batches]
        self.num_batches -= validation_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0



def visualize_result():
    import pandas as pd
    import matplotlib.pyplot as plt

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    files = [('GridGRU, 3 layers', 'save_gridgru3layers/log.csv'),
             # ('GridGRU, 6 layers', 'save_gridgru6layers/log.csv'),
             ('GridLSTM, 3 layers', 'save_gridlstm3layers/log.csv'),
             ('GridLSTM, 6 layers', 'save_gridlstm6layers/log.csv'),
             ('Stacked GRU, 3 layers', 'save_gru3layers/log.csv'),
             # ('Stacked GRU, 6 layers', 'save_gru6layers/log.csv'),
             ('Stacked LSTM, 3 layers', 'save_lstm3layers/log.csv'),
             ('Stacked LSTM, 6 layers', 'save_lstm6layers/log.csv'),
             ('Stacked RNN, 3 layers', 'save_rnn3layers/log.csv'),
             ('Stacked RNN, 6 layers', 'save_rnn6layers/log.csv')]
             
    file1= './save/tinyshakespeare/{}'     
    for i, (k, v) in enumerate(files):
        train_loss = pd.read_csv(file1.format(v)).groupby('epoch').mean()['train_loss']
        plt.plot(train_loss.index.tolist(), train_loss.tolist(), label=k, lw=2, color=tableau20[i*2])
    plt.leg()
    plt.xlabel('Epochs')
    plt.ylabel('Average training loss')
    plt.show()






####################################################################################       
####################################################################################   
# sample.py
import argparse, cPickle, os, tensorflow as tf
from model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,  help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ', help='prime text')
    args = parser.parse_args()
    sample(args)


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl')) as f:      saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl')) as f: chars, vocab = cPickle.load(f)
        
        
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print model.sample(sess, chars, vocab, args.n, args.prime)


if __name__ == '__main__':
    main()

        
        
        
        