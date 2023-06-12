## Dataset patern

def plot_predictons(train_data = X_train,
                    train_labels =  y_train,
                    test_data = X_test,
                    test_labels = y_test,
                    predictions =None):
                    
  plt.figure(figsize=(5,5))

  plt.scatter(train_data, train_labels, c='b', s=4, label="Training data")    # Training data in blue
  plt.scatter(test_data, test_labels, c='r', s=4, label="Testing data")       # test data in red

  # are there predictions?
  if predictions is not None:
   plt.scatter(test_data, predictions, c='g', s=4, label="predictions")

  plt.legend(prop={'size':12});
  plot_predictons()


------------------------------------------------
## To check output before training

with torch.inference_mode():
 y_guess = model_0(X_test)
plot_predictons(predictions = y_guess)

------------------------------------------------
## To check output after training

plot_predictons(predictions = ytest_preds)

------------------------------------------------
## Slope in loss function. 

plt.plot(epoch_c, loss_v, label = 'Train Loss')
plt.plot(epoch_c, testl_v, label = 'Test Loss')
plt.title('T & T curves')
plt.ylabel('Loss')
plt.xlabel('Epocs')
plt.legend();


