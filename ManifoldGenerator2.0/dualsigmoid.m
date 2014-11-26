function a = dualsigmoid(z,hw)

a = sigmoid(z-hw) + sigmoid(-(z+hw));
