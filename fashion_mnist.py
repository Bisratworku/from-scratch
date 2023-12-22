from scratch import * 
model = Model()
model.add(Layer_Dense(28 * 28 , 64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.5))
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 10))
model.add(Activation_Sigmoid())

model.set(loss = Loss_CategoricalCrossentropy(),
          optimizer = Optimizer_Adam(learning_rate = 0.005, decay = 1e-3),
          accuracy= Accuracy_Categorical())
model.finalize()
