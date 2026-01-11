"""
How do we make a LR scheduler that works at the iteration level instead of the
epoch level? Is this even a good idea?


References:
    # Shows how PJR divides learning rate by batch size
    https://github.com/pjreddie/darknet/blob/e7405b513dc69f17e9e75e8d306d22f2e08c1489/src/convolutional_layer.c#L538

    # Shows how burn_in LR is computed
    https://github.com/pjreddie/darknet/blob/d3828827e70b293a3045a1eb80bfb4026095b87b/src/network.c#L95

    if (batch_num < net->burn_in) return net->learning_rate * pow((float)batch_num / net->burn_in, net->power);


base_lr = .01

burn_in = 1000
power = 4

base_lr * pow(batch_num / burn_in, power)


epoch_frac = np.linspace(0, 1, 10)

base_lr * pow(epoch_frac, power)


"""
