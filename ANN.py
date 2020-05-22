import numpy

def sigmoid(inpt):
    return 1.0/(1.0+numpy.exp(-1*inpt))

def relu(inpt):
    result = inpt
    result[inpt<0] = 0
    return result

def MSE(y, t):
    return 0.5 * (y - t)**2

def predict_outputs(weights_mat, data_inputs, data_outputs, activation="relu"):
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    losses = numpy.zeros(shape=(data_inputs.shape[0]))
    for sample_idx in range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        layer_ix=0
        for curr_weights in weights_mat:
            r1 = numpy.matmul(r1, curr_weights)
            if layer_ix==2:
                pass
            else:
                if activation == "relu":
                    r1 = relu(r1)
                elif activation == "sigmoid":
                    r1 = sigmoid(r1)
            layer_ix+=1
        losses[sample_idx] = MSE(r1,data_outputs[sample_idx])
        predictions[sample_idx] = r1
    loss_value=numpy.sum(losses)/data_inputs.shape[0]
    return loss_value,predictions
    
def fitness(weights_mat, data_inputs, data_outputs, activation="relu"):
    losses = numpy.empty(shape=(weights_mat.shape[0]))
    for sol_idx in range(weights_mat.shape[0]):
        curr_sol_mat = weights_mat[sol_idx, :]
        losses[sol_idx],_= predict_outputs(curr_sol_mat, data_inputs, data_outputs, activation=activation)
    return losses

