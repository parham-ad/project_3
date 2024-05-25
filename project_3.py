#Import the librarys
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import sleep

#Get data from dataset.csv
data = pd.read_csv('dataset.csv')

#Split the data into training and testing 
x = data[['x']]
y = data['y']
X_train , X_test ,y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state = 42)

#Creat and train a liner regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Use the trained models to make predictions method on the test data
y_pred = model.predict(X_test)

#Asses the model's preformance using metrics such mean squared error
mse_er = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse_er}')

#creat a plot :
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
plt.title('Linear Regression For Dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

#Ask questions for the chart
while True:
    q = input("Do you need plot(y/n)? : ")
    if q == 'y' :
        plt.show()
        break
    
    elif q == 'n' :
        break
    else :
        print("the command is not correct! please try again")

#The end
print("parham code")
sleep(5)
exit()


