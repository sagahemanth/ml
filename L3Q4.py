import pandas as pd
import statistics
import matplotlib.pyplot as plt

excel_file_path = 'Lab Session1 Data.xlsx'
df = pd.read_excel(excel_file_path, sheet_name='IRCTC Stock Price')

price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print(f"Mean of Price: {price_mean}\n")
print(f"Variance of Price: {price_variance}\n")

wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price on Wednesdays: {wednesday_mean}\n")


april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price in April: {april_mean}\n")


loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}\n")
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}\n")
conditional_profit_probability = wednesday_profit_probability / loss_probability
print(f"Conditional Probability of making profit, given today is Wednesday: {conditional_profit_probability}\n")
day=['Mon','Tue','Wed','Thu','Fri']
day1=[]
chg1=[]
for i in day:
    for j in range(2,len(df['Day'])):
        if i==df.loc[j,'Day']:
            day1.append(i)
            chg1.append(df.loc[j,'Chg%'])
plt.scatter(day1, chg1)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% against the day of the week')
plt.show()