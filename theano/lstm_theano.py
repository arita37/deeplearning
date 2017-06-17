# -*- coding: utf-8 -*-
"""
LSTM in Theano

http://deeplearning.net/tutorial/code/lstm.py
"""


################################################################################
############## lSTM Grid in Numpy with batch mode
#   This is a batched LSTM forward and backward pass
import numpy as np;  import code

class gridLSTM :
  @staticmethod
  def init(input_size, hidden_size, grid_size fancy_forget_bias_init = 3) :
    """    Initialize parameters of the LSTM Grid """
    
    HH= np.zeros((grid_size,4*hidden_size))    #hidden layers shared to all
    
    MM= np.zeros((grid_size, hidden_size))    #Memory cell grid
        
    #Weights: Ngrid x (Input+d) x 4d +1 for the biases,1st row of WLSTM
    WLSTM= np.random.randn(grid_size, input_size+hidden_size + 1,
                           4*hidden_size)/np.sqrt(grid_size+input_size + hidden_size)
    
    # initialize biases to zero
    WLSTM[:,0,:] = 0 
    if fancy_forget_bias_init != 0:  WLSTM[:,0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return HH, MM, WLSTM


  @staticmethod
  def forward(X, HH, MM, WW, ngrid, c0 = None, h0 = None):

   for n in range(0,ngrid):
       Hout, Ct, Houtt, cache= LSTM.forward2(X, HH, MM[n,:], WW[n,:], c0 = Ct, h0 = ht)

   return Hout
     
     
# Grid Dimension: 2,   Dim_1: Time,   Dim_2: Depth
#   (Xi, Yi),  Xi: 1..d ,   m samples
#   h_t: 1 x 4d   (h_dim1)
#   h_u: 1 x 4d   (h_dim2)
#   prev_m, next_h, next_m: 1 x d



class LSTM:
  
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
    """ 
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5,)
    """
    # +1 for the biases, which will be the first row of WLSTM
    WLSTM= np.random.randn(input_size+hidden_size + 1, 4*hidden_size)/np.sqrt(input_size + hidden_size)
    WLSTM[0,:] = 0 # initialize biases to zero
    if fancy_forget_bias_init != 0:
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return WLSTM
  
  
  @staticmethod
  def forward(X, WLSTM, c0 = None, h0 = None):
    #X should be of shape (n,b,input_size), where n=  t lengthofsequence, b= batch size
    n,b,input_size = X.shape
    d = WLSTM.shape[1]/4 # hidden size
    if c0 is None: c0 = np.zeros((b,d))
    if h0 is None: h0 = np.zeros((b,d))
    
    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content
    
    
    
    for t in xrange(n):
      # concat [x,h] as input to the LSTM
      prevh = Hout[t-1] if t > 0 else h0
      Hin[t,:,0] = 1 # bias
      Hin[t,:,1:input_size+1] = X[t]
      Hin[t,:,input_size+1:] = prevh

      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM)
      
      # non-linearities
      IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh

      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
      Ct[t] = np.tanh(C[t])
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]

    cache = {}
    cache['WLSTM'] = WLSTM;    cache['Hout'] = Hout;    cache['IFOGf'] = IFOGf;    
    cache['IFOG'] = IFOG;    cache['C'] = C;    cache['Ct'] = Ct;    
    cache['Hin'] = Hin; cache['c0'] = c0;    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache

 
  @staticmethod
  def backward(dHout_in, cache, dcn = None, dhn = None): 
    WLSTM = cache['WLSTM'];    Hout = cache['Hout']
    IFOGf = cache['IFOGf'];    IFOG = cache['IFOG']
    C = cache['C'];    Ct = cache['Ct'];    Hin = cache['Hin'];    c0 = cache['c0'];    
    h0 = cache['h0']
  
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias
 
    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape);    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape);    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape);    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d));    dc0 = np.zeros((b, d))
    
    #From Output Value (copy to prevent side effect)  --> dWeight
    dHout = dHout_in.copy() 

    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
        
    for t in reversed(xrange(n)): #from T....to 1
    
      #Memory Cell
      tanhCt = Ct[t]
      dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
      # backprop tanh non-linearity first then continue backprop
      dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
 
      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
      else:
        dIFOGf[t,:,d:2*d] = c0 * dC[t]
        dc0 = IFOGf[t,:,d:2*d] * dC[t]
        
      dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
      dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
      
      
      # backprop activation functions
      dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]

 
      # backprop initial matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())

 
      # Get Xt,Ht-1 from Hin
      dX[t] = dHin[t,:,1:input_size+1]  #Xt=(0..size)     h0=(size+1..size+d)
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:]
      else:
        dh0 += dHin[t,:,input_size+1:]

 
    return dX, dWLSTM, dc0, dh0



# -------------------
# TEST CASES
def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in xrange(n)]
  Hcat = np.zeros((n,b,d))
  for t in xrange(n):
    xt = X[t:t+1]
    _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
    caches[t] = cache
    Hcat[t] = hprev

  # sanity check: perform batch forward to check that we get the same thing
  H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

  # eval loss
  wrand = np.random.randn(*Hcat.shape)
  loss = np.sum(Hcat * wrand)
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(WLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(xrange(n)):
    dht = dH[t].reshape(1, b, d)
    dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print 'Making sure batched version agrees with sequential version: (should all be True)'
  print np.allclose(BdX, dX)
  print np.allclose(BdWLSTM, dWLSTM)
  print np.allclose(Bdc0, dc0)
  print np.allclose(Bdh0, dh0)
  

def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

  def fwd():
    h,_,_,_ = LSTM.forward(X, WLSTM, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
  for j in xrange(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in xrange(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0 # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0 # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


if __name__ == "__main__":

  checkSequentialMatchesBatch()
  raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  print 'every line should start with OK. Have a nice day!'























################################################################################
############## lSTM Grid in Numpy

#import 'nn'
#import 'nngraph'
'''
  Grid Dimension: 2,   Dim_1: Time,   Dim_2: Depth
  
   (Xi, Yi),  Xi: 1..d ,   m samples
   h_t: 1 x 4d   (h_dim1)
   h_u: 1 x 4d   (h_dim2)
   prev_m, next_h, next_m: 1 x d
   
   
   
  Single LSTM (H,m, W)
    Xi: Input  (1 x p)
    projMatrix: (d x p) : projection of the Xi into d 
    [Wu, Wf, Wo, Wc]= W    #Wu_dim=   (d * 2d)  (1_d for hidden, 1_d for memory_cell)
    H= [projMatrix * Xi, h]

    gu = Wu x H
    gf = Wf x H
    go = Wo x H
    gc = Wc x H 
    m' = gf x m  + gu x gc
    h'= tanh(go x m')
    
    (h',m') = LSTM(H,m,W)

   
   
   Grid_LSTM   
   
     H= [h1, .., hN]   concatene hidde layers from N dimensions
     m_1, ..., m_N : N memory cells

     For all grid dimension i=1..N
         W_i= [Wu_i, Wf_i, Wo_i, Wc_i]   #Weight for layers
         
         Hpriority_i= [h1',...,h'_i-1, hi,..,hN]  #Use update values       
         
        (h_i', m_i')=  LSTM(Hpriority__i, m_i, W_i)

  This is called once per dimension inside grid LSTM block to create gated
  update of dimension's hidden state and memory cell.
  It takes:
    h_t and h_d, hidden states from time_dim and depth_dim
    prev_m  dimension's previous memory cell.
  
  It returns: 
    next_c, next_h along dimension, using standard lstm gated update, 
    conditioned on concatenated time (dim 1)  and  depth (dim 2) hidden states .
  
'''


def lstm(h_t, h_d, prev_m, rnn_size) :
  # h_t: 1 x 4d   input to hidden, temporal  
  # h_u: 1 x 4d   hidden to hidden,  depth
  # prev_m, next_h, next_m: 1 x d

  input_sum= np.concatenate((h_t,h_d), axis=0)  #Stack into Grid Format
  
  reshaped = np.reshape(input_sum, (4, rnn_size))
  wuh =  reshaped[0]
  wfh =  reshaped[1]
  woh =  reshaped[2]
  wch =  reshaped[3]

  # Decode gates
  gu_ingate = sigmoid(wuh)        # gu = Wu x H
  gf_forget_gate = sigmoid(wfh)    # gf = Wf x H
  go_out_gate = sigmoid(woh)       # go = Wo x H
  gc_in_transform = np.tanh(wch)   # gc = Wc x H decode write inputs

  # Memory cell update  m' = gf X m  + gu X gc
  next_m = np.dot(gf_forget_gate, prev_m) +  np.dot(gu_in_gate, gc_in_transform)
     
  # Hidden layer update:   H'= tanh(go x m')
  next_h = np.tanh(np.dot(out_gate, next_m))
  
  return next_m, next_h 



'''GridLSTM:
 1) Map input x into memory and hidden cells m(1), h(1) along depth dimension.
 2) Concatenate previous hidden states from time and depth dimensions, [h(1), h(2)] into H.
 3) Forward time LSTM, LSTM_2(H) -> h(2)', m(2)'.
 4) Concatenate transformed h(2)' and h(1) into H' = [h(1), h(2)']
 5) Forward depth LSTM, LSTM_1(H') -> h(1)', m(1)'
 6) Either repeat 2-5 for another layer or map h(1)', final hidden state along depth 
       dimension, to a character prediction.
       
       
       
       
       
       
'''
GridLSTM = []
def GridLSTM_grid_lstm(input_size, rnn_size, n, dropout, should_tie_weights) :
  # n : nb of depth layers,  rnn_size : d  , input_size
  d= rnn_size

  #---------------------------Input------------------------------------------
  # md0, hd0  :  Depth dimension, memory, hidden
  # mt_1, ht_1, ..... mt_n, ht_n  : Time dimension: memory, hidden

  inputs = []   # 2*n+2 inputs
  
  #Initial Depth dimension
  inputs.append(np.identity(d)) # initial memory cell
  inputs.append(np.identity(d)) # initial hidden cell
  
  #Initial Time Dimension, all depth layers 1..n
  for L  in range(1,n)  :         
    inputs.append(np.identity(d)) # initial time memory cell   # prev_m[L] 
    inputs.append(np.identity(d)) # initial time hidden cell   # prev_h[L] for time dimension
  

  if should_tie_weights == 1  :    #  shared_weights
      shared_weights = [nn.Linear(rnn_size, 4 * rnn_size), nn.Linear(rnn_size, 4 * rnn_size)] 


  #---------------------------Output------------------------------------------
  outputs_t = [] # Time dimension:  Outputs being handed to next time step along 
  outputs_d = [] # Depth dimension: Outputs being handed from one layer to next along 

  for L in range(1,n)  : 
    #Time dimension: t-1 time step, layer L, 
    prev_m_t = inputs[L*2+1];     prev_h_t = inputs[L*2+2]

    #Depth dimension:  take previous layer,  memory cell  m_t, hidden: h_t 
    if L == 1  :   # 1st  layer
      prev_m_d = inputs[1]              # input_m_d: initial memory cell, just a zero vec.
      prev_h_d = nn.LookupTable(input_size, rnn_size, inputs[2])  #map a char with lookup table
      
    else :          # Layers 2...N, Take hidden and memory cell from Updated layer (L-1)
      prev_m_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]

      if dropout > 0  :  #on hidden Depth layer
          prev_h_d = Dropout(dropout, prev_h_d, annotate, name='drop_', L) 


    #----------------Dim_1: Time dim calculation---------------------------------------
    # Time dimension, Evaluate input sums at once for efficiency
    #TimeWeight, DepthWeight, Memory ----->  (memory_time,  hidden_time states)

    t2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}  #Time_Weight
    d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}  #Depth_Weight
    next_m_t, next_h_t = lstm(t2h_t, d2h_t, prev_m_t, rnn_size)
    outputs_t.append(next_m_t);  outputs_t.append(next_h_t)
    # See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
    # weights along temporal dimension are already tied (cloned many times in train.lua)


    #----------------Dim_2: Depth dim calculation--------------------------------------
    # [Wu, ..,Wc] x H    Evaluate input sums at once for efficiency


    # "Priority Dimensions" use new]hidden_time :   next_h_t instead of previous temporal hidden  
    # implements Section 3.2, 
    # Scale H_ to (d, 4*d) size 
    t2h_d = nn.Linear(rnn_size, 4 * rnn_size)(next_h_t):annotate{name='i2h_'..L}  #NEW Time_Weight
    d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}  #OLD Depth_Weight

    # Tie weights along depth dimension. Having invariance in computation is important
    if should_tie_weights == 1  : 
      print("tying weights along depth dimension")
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    
    next_m_d, next_h_d = lstm(t2h_d, d2h_d, prev_m_d, rnn_size)
    outputs_d.append(next_m_d);  outputs_d.append(next_h_d)  #Update  output


    # along depth appears be critical to solve 15 digit addition problem w/ high accy.



  #----------------Output softlayer probability---------------------------------- 
  # set up decoder on last layer
  top_h = outputs_d[-1] # last one outputs_d]  as output
  if dropout > 0  :  top_h = Dropout(dropout, top_h) :
      
  proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  
  logsoft = LogSoftMax(proj)
  outputs_t.append(logsoft)

  return outputs_t
  

#  return nn.gModule(inputs, outputs_t)
#return GridLSTM

module = Linear(inputDimension,outputDimension)

Applies a linear transformation to the incoming data, i.e. //y= Ax+b//. The input tensor given in forward(input) must be either a vector (1D tensor) or matrix (2D tensor). If the input is a matrix, then each row is assumed to be an input sample of given batch.

You can create a layer in the following way:

 module= nn.Linear(10,5)  -- 10 inputs, 5 outputs










"""
This is a batched LSTM forward and backward pass
"""
import numpy as np
import code  #Interactive code 

class LSTM:
  
  @staticmethod
  def init(input_size, hidden_size, fancy_forget_bias_init = 3):
    """ 
    Initialize parameters of the LSTM (both weights and biases in one matrix) 
    One might way to have a positive fancy_forget_bias_init number (e.g. maybe even up to 5, in some papers)
    """
    # +1 for the biases, which will be the first row of WLSTM
    WLSTM = np.random.randn(input_size + hidden_size + 1, 4 * hidden_size) / np.sqrt(input_size + hidden_size)
    WLSTM[0,:] = 0 # initialize biases to zero
    if fancy_forget_bias_init != 0:
      # forget gates get little bit negative bias initially to encourage them to be turned off
      # remember that due to Xavier initialization above, the raw output activations from gates before
      # nonlinearity are zero mean and on order of standard deviation ~1
      WLSTM[0,hidden_size:2*hidden_size] = fancy_forget_bias_init
    return WLSTM
  
  @staticmethod
  def forward(X, WLSTM, c0 = None, h0 = None):
    """
    X should be of shape (n,b,input_size), where n = length of sequence, b = batch size
    """
    n,b,input_size = X.shape
    d = WLSTM.shape[1]/4 # hidden size
    if c0 is None: c0 = np.zeros((b,d))
    if h0 is None: h0 = np.zeros((b,d))
    
    # Perform the LSTM forward pass with X as the input
    xphpb = WLSTM.shape[0] # x plus h plus bias, lol
    Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
    Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
    IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
    IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
    C = np.zeros((n, b, d)) # cell content
    Ct = np.zeros((n, b, d)) # tanh of cell content
    for t in xrange(n):
      # concat [x,h] as input to the LSTM
      prevh = Hout[t-1] if t > 0 else h0
      Hin[t,:,0] = 1 # bias
      Hin[t,:,1:input_size+1] = X[t]
      Hin[t,:,input_size+1:] = prevh
      # compute all gate activations. dots: (most work is this line)
      IFOG[t] = Hin[t].dot(WLSTM)
      # non-linearities
      IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
      IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
      # compute the cell activation
      prevc = C[t-1] if t > 0 else c0
      C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
      Ct[t] = np.tanh(C[t])
      Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]

    cache = {}
    cache['WLSTM'] = WLSTM
    cache['Hout'] = Hout
    cache['IFOGf'] = IFOGf
    cache['IFOG'] = IFOG
    cache['C'] = C
    cache['Ct'] = Ct
    cache['Hin'] = Hin
    cache['c0'] = c0
    cache['h0'] = h0

    # return C[t], as well so we can continue LSTM with prev state init if needed
    return Hout, C[t], Hout[t], cache
  
  @staticmethod
  def backward(dHout_in, cache, dcn = None, dhn = None): 

    WLSTM = cache['WLSTM']
    Hout = cache['Hout']
    IFOGf = cache['IFOGf']
    IFOG = cache['IFOG']
    C = cache['C']
    Ct = cache['Ct']
    Hin = cache['Hin']
    c0 = cache['c0']
    h0 = cache['h0']
    n,b,d = Hout.shape
    input_size = WLSTM.shape[0] - d - 1 # -1 due to bias
 
    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape)
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros((n,b,input_size))
    dh0 = np.zeros((b, d))
    dc0 = np.zeros((b, d))
    dHout = dHout_in.copy() # make a copy so we don't have any funny side effects
    if dcn is not None: dC[n-1] += dcn.copy() # carry over gradients from later
    if dhn is not None: dHout[n-1] += dhn.copy()
    for t in reversed(xrange(n)):
 
      tanhCt = Ct[t]
      dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
      # backprop tanh non-linearity first then continue backprop
      dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])
 
      if t > 0:
        dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
        dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
      else:
        dIFOGf[t,:,d:2*d] = c0 * dC[t]
        dc0 = IFOGf[t,:,d:2*d] * dC[t]
      dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
      dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]
      
      # backprop activation functions
      dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
      y = IFOGf[t,:,:3*d]
      dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]
 
      # backprop matrix multiply
      dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())
 
      # backprop the identity transforms into Hin
      dX[t] = dHin[t,:,1:input_size+1]
      if t > 0:
        dHout[t-1,:] += dHin[t,:,input_size+1:]
      else:
        dh0 += dHin[t,:,input_size+1:]
 
    return dX, dWLSTM, dc0, dh0



# -------------------
# TEST CASES
# -------------------

def checkSequentialMatchesBatch():
  """ check LSTM I/O forward/backward interactions """

  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # sequential forward
  cprev = c0
  hprev = h0
  caches = [{} for t in xrange(n)]
  Hcat = np.zeros((n,b,d))
  for t in xrange(n):
    xt = X[t:t+1]
    _, cprev, hprev, cache = LSTM.forward(xt, WLSTM, cprev, hprev)
    caches[t] = cache
    Hcat[t] = hprev

  # sanity check: perform batch forward to check that we get the same thing
  H, _, _, batch_cache = LSTM.forward(X, WLSTM, c0, h0)
  assert np.allclose(H, Hcat), 'Sequential and Batch forward don''t match!'

  # eval loss
  wrand = np.random.randn(*Hcat.shape)
  loss = np.sum(Hcat * wrand)
  dH = wrand

  # get the batched version gradients
  BdX, BdWLSTM, Bdc0, Bdh0 = LSTM.backward(dH, batch_cache)

  # now perform sequential backward
  dX = np.zeros_like(X)
  dWLSTM = np.zeros_like(WLSTM)
  dc0 = np.zeros_like(c0)
  dh0 = np.zeros_like(h0)
  dcnext = None
  dhnext = None
  for t in reversed(xrange(n)):
    dht = dH[t].reshape(1, b, d)
    dx, dWLSTMt, dcprev, dhprev = LSTM.backward(dht, caches[t], dcnext, dhnext)
    dhnext = dhprev
    dcnext = dcprev

    dWLSTM += dWLSTMt # accumulate LSTM gradient
    dX[t] = dx[0]
    if t == 0:
      dc0 = dcprev
      dh0 = dhprev

  # and make sure the gradients match
  print 'Making sure batched version agrees with sequential version: (should all be True)'
  print np.allclose(BdX, dX)
  print np.allclose(BdWLSTM, dWLSTM)
  print np.allclose(Bdc0, dc0)
  print np.allclose(Bdh0, dh0)
  

def checkBatchGradient():
  """ check that the batch gradient is correct """

  # lets gradient check this beast
  n,b,d = (5, 3, 4) # sequence length, batch size, hidden size
  input_size = 10
  WLSTM = LSTM.init(input_size, d) # input size, hidden size
  X = np.random.randn(n,b,input_size)
  h0 = np.random.randn(b,d)
  c0 = np.random.randn(b,d)

  # batch forward backward
  H, Ct, Ht, cache = LSTM.forward(X, WLSTM, c0, h0)
  wrand = np.random.randn(*H.shape)
  loss = np.sum(H * wrand) # weighted sum is a nice hash to use I think
  dH = wrand
  dX, dWLSTM, dc0, dh0 = LSTM.backward(dH, cache)

  def fwd():
    h,_,_,_ = LSTM.forward(X, WLSTM, c0, h0)
    return np.sum(h * wrand)

  # now gradient check all
  delta = 1e-5
  rel_error_thr_warning = 1e-2
  rel_error_thr_error = 1
  tocheck = [X, WLSTM, c0, h0]
  grads_analytic = [dX, dWLSTM, dc0, dh0]
  names = ['X', 'WLSTM', 'c0', 'h0']
  for j in xrange(len(tocheck)):
    mat = tocheck[j]
    dmat = grads_analytic[j]
    name = names[j]
    # gradcheck
    for i in xrange(mat.size):
      old_val = mat.flat[i]
      mat.flat[i] = old_val + delta
      loss0 = fwd()
      mat.flat[i] = old_val - delta
      loss1 = fwd()
      mat.flat[i] = old_val

      grad_analytic = dmat.flat[i]
      grad_numerical = (loss0 - loss1) / (2 * delta)

      if grad_numerical == 0 and grad_analytic == 0:
        rel_error = 0 # both are zero, OK.
        status = 'OK'
      elif abs(grad_numerical) < 1e-7 and abs(grad_analytic) < 1e-7:
        rel_error = 0 # not enough precision to check this
        status = 'VAL SMALL WARNING'
      else:
        rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
        status = 'OK'
        if rel_error > rel_error_thr_warning: status = 'WARNING'
        if rel_error > rel_error_thr_error: status = '!!!!! NOTOK'

      # print stats
      print '%s checking param %s index %s (val = %+8f), analytic = %+8f, numerical = %+8f, relative error = %+8f' \
            % (status, name, `np.unravel_index(i, mat.shape)`, old_val, grad_analytic, grad_numerical, rel_error)


if __name__ == "__main__":

  checkSequentialMatchesBatch()
  raw_input('check OK, press key to continue to gradient check')
  checkBatchGradient()
  print 'every line should start with OK. Have a nice day!'
















###################################################################################
##############     RNN in Numpy Python                      #######################




#Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
import numpy as np


# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }



# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll RNN for
learning_rate = 1e-1




# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias



#  returns loss, gradients on model parameters, and last hidden state
def lossFun(inputs, targets, hprev):
  """ inputs,targets are both list of integers.
      hprev is Hx1 array of initial hidden state
      returns loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
    
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]


def sample(h, seed_ix, n):
  """ sample sequence of integers from model 
      h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes



#------Generate sentence from DNN data-------------------------------
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from model now and then
  if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

  # forward seq_length characters through net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 
  
  
  
  
  








####################################################################################
####################################################################################
####################Theano based LSTM ###################################

#Build tweet sentiment analyzer
from __future__ import print_function
import six.moves.cPickle as pickle
from collections import OrderedDict
import sys, time,  util

import numpy as np, theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import theano_imdb as imdb




datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

def np_floatX(data):
    return np.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """ Used to shuffle dataset at each iteration.    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """ When we reload model. Needed for GPU stuff. """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """ When we pickle model. Needed for GPU stuff. """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise, (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1, dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """    Global (not LSTM) parameter. For embeding and classifier.    """
    params = OrderedDict()

    # embedding: SizeAlphabet x NbhiddenNeurons
    randn = np.random.rand(options['n_words'], options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options, params, prefix=options['encoder'])
    
    # classifier
    params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                         options['ydim']).astype(config.floatX)
    params['b'] = np.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp: raise Warning('%s is not in archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim): #SVD of random matrix for initlialization
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """ Init LSTM parameter:see: init_params """
    W = np.concatenate([   ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    
    U = np.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    
    b = np.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params



def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:   n_samples = state_below.shape[1]
    else:                       n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):  #Memory cell definition for LSTM
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                   sequences=[mask, state_below],
                   outputs_info=[tensor.alloc(np_floatX(0.),  n_samples, dim_proj),
                   tensor.alloc(np_floatX(0.),  n_samples,  dim_proj)],
                                name=_p(prefix, '_layers'), n_steps=nsteps)
    return rval[0]







import 'nn'
import 'nngraph'
'''
  This is called once per dimension inside grid LSTM block to create gated
  update of dimension's hidden state and memory cell.
  It takes h_t and h_d, hidden states from temporal and 
  depth dimensions respectively, as well as prev_c,  dimension's previous memory cell.
  
  It returns next_c, next_h along dimension, using standard lstm gated update, 
  conditioned on concatenated time and  depth hidden states.
'''
def lstm(h_t, h_d, prev_c, rnn_size)
  all_input_sums = nn.CAddTable()({h_t, h_d})
  reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
  n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
  
  # decode gates
  in_gate = nn.Sigmoid()(n1)
  forget_gate = nn.Sigmoid()(n2)
  out_gate = nn.Sigmoid()(n3)

  # decode write inputs
  in_transform = nn.Tanh()(n4)

  # perform LSTM update
  next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
    })

  # gated cells form output
  next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h





'''
  GridLSTM:
    1) Map input x into memory and hidden cells m(1), h(1) along depth dimension.
    2) Concatenate previous hidden states from time and depth dimensions, [h(1), h(2)] into H.
    3) Forward time LSTM, LSTM_2(H) -> h(2)', m(2)'.
    4) Concatenate transformed h(2)' and h(1) into H' = [h(1), h(2)']
    5) Forward depth LSTM, LSTM_1(H') -> h(1)', m(1)'
    6) Either repeat 2-5 for another layer or map h(1)', final hidden state along depth 
       dimension, to character prediction.
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

  outputs_t = {} # Outputs being handed to next time step along time dimension
  outputs_d = {} # Outputs being handed from one layer to next along depth dimension

  for L in range(1,n)  : 
    # Take hidden and memory cell from previous time steps
    prev_m_t = inputs[L*2+1]
    prev_h_t = inputs[L*2+2]

    if L == 1  : 
      # We're in first layer
      prev_m_d = inputs[1] # input_m_d: starting depth dimension memory cell, just zero vec.
      prev_h_d = nn.LookupTable(input_size, rnn_size)(inputs[2]) 
      # input_h_d: starting depth dim hidden state. map char into hidden space via lookup table
    else 
      # We're in higher layers 2...N
      # Take hidden and memory cell from layers below
      prev_m_d = outputs_d[((L-1)*2)-1]
      prev_h_d = outputs_d[((L-1)*2)]
      if dropout > 0  :  prev_h_d = nn.Dropout(dropout)(prev_h_d):annotate{name='drop_' .. L}  # apply dropout, if any
    

    # Evaluate input sums at once for efficiency
    t2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_t):annotate{name='i2h_'..L}
    d2h_t = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}
    
    # Get transformed memory and hidden states pointing in time direction first
    next_m_t, next_h_t = lstm(t2h_t, d2h_t, prev_m_t, rnn_size)

    # Pass memory cell and hidden state to next timestep
    table.insert(outputs_t, next_m_t)
    table.insert(outputs_t, next_h_t)

    # Evaluate input sums at once for efficiency
    t2h_d = nn.Linear(rnn_size, 4 * rnn_size)(next_h_t):annotate{name='i2h_'..L}
    d2h_d = nn.Linear(rnn_size, 4 * rnn_size)(prev_h_d):annotate{name='h2h_'..L}

    # See section 3.5, "Weight Sharing" of http://arxiv.org/pdf/1507.01526.pdf
    # weights along temporal dimension are already tied (cloned many times in train.lua)
    # Here we can tie weights along depth dimension. Having invariance in computation
    # along depth appears to be critical to solving 15 digit addition problem w/ high accy.
    # See fig 4. to compare tied vs untied grid lstms on this task.
    if should_tie_weights == 1  : 
      print("tying weights along depth dimension")
      t2h_d.data.module:share(shared_weights[1], 'weight', 'bias', 'gradWeight', 'gradBias')
      d2h_d.data.module:share(shared_weights[2], 'weight', 'bias', 'gradWeight', 'gradBias')
    
    
    # Create lstm gated update pointing in depth direction.
    # We 'prioritize' depth dimension by using updated temporal hidden state as input
    # instead of previous temporal hidden state. This implements Section 3.2, "Priority Dimensions"
    next_m_d, next_h_d = lstm(t2h_d, d2h_d, prev_m_d, rnn_size)

    # Pass depth dimension memory cell and hidden state to layer above
    table.insert(outputs_d, next_m_d)
    table.insert(outputs_d, next_h_d)
  

  # set up decoder
  top_h = outputs_d[#outputs_d]
  if dropout > 0  :  top_h = nn.Dropout(dropout)(top_h) 
  proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs_t, logsoft)

  return nn.gModule(inputs, outputs_t)


return GridLSTM




















# ff: Feed Forward (normal neural net), only useful to put after lstm, before classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}



def build_model(tparams, options):
    trng = RandomStreams(SEED)


    use_noise = theano.shared(np_floatX(0.))       # Used for dropout.

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    # TimeStep x Nsamples  Neural Network
    n_timesteps = x.shape[0];    n_samples = x.shape[1]
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,  n_samples,  options['dim_proj']])
    
    # 
    proj = get_layer(options['encoder'])[1](tparams, emb, options, prefix=options['encoder'],
                                            mask=mask)
    if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]
        
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)


    #Prediction output
    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
#    if pred.dtype == 'float16':  off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost


def pred_probs(f_pred_prob, prepare_data, data, iterator, verbose=False):
    """ If you want to use trained model, this is useful to compute
    probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = np.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """ Just compute error,  f_pred: Theano fct computing prediction
        prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  np.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x, mask)
        targets = np.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - np_floatX(valid_err) / len(data[0])

    return valid_err



def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent  :note: more complicated version of
    sgd then needed.  This is done like that for adadelta and rmsprop.
    """
    # New set of shared variable that will contain gradient for mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function computes gradients for mini-batch, but  not updates weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates weights from previously computed gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable,  Initial learning rate
    tpramas: Theano SharedVariable, Model parameters
    grads: Theano variable, Gradients of cost w.r.t to parameres
    x: Theano variable,   Model inputs
    mask: Theano variable,  Sequence mask
    y: Theano variable, Targets
    cost: Theano variable, Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.
      [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * np_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """  variant of  SGD that scales step size by running average of recent step norms.

    Parameters
    lr : Theano SharedVariable, Initial learning rate
    tpramas: Theano SharedVariable, Model parameters
    grads: Theano variable, Gradients of cost w.r.t to parameres
    x: Theano variable, Model inputs
    mask: Theano variable, Sequence mask
    y: Theano variable, Targets
    cost: Theano variable, Objective fucntion to minimize

    Notes
       [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * np_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * np_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * np_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update




def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # maximum number of epoch to run
    dispFreq=10,  # Display to stdout training progress every N updates
    decay_c=0.,  # Weight decay for classifier applied to U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # best model will be saved there
    validFreq=370,  # Compute validation error after this number of update.
    saveFreq=1110,  # Save parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=16,  # batch size during training.
    valid_batch_size=64,  # batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need bigger model.
    reload_model=None,  # Path to saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print("model options", model_options)

    load_data, prepare_data = get_dataset(dataset)

    print('Loading data')
    train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
                                   maxlen=maxlen)
    if test_size > 0:
        # test set is sorted by size, but we want to keep random
        # size example.  So we must select random selection of the
        # examples.
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = np.max(train[1]) + 1

    model_options['ydim'] = ydim

    print('Building model')
    # This create initial parameters as np ndarrays.
    # Dict name (string) -> np ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(np_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:   validFreq = len(train[0]) // batch_size
    if saveFreq == -1:    saveFreq = len(train[0]) // batch_size

    uidx = 0  # number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t]for t in train_index]

                # Get data in np.ndarray format
                # This swap axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)
    return train_err, valid_err, test_err


# See function train for all possible parameter and there definition.
#train_lstm( max_epochs=1,test_size=5, )
    
    
    
    
    
    