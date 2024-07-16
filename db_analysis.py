import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
        
class DataAnalysis:
    """
    Class for completing data analysis on the database
    """
    def __init__(self, df):
        self.df = df

    def state_of_loans(self):
        """
        Plots the total amount of loans that are currently paid off
        """
        loan = sum(self.df['total_rec_prncp'])/sum(self.df['loan_amount'])*100
        plt.pie([loan,100-loan], labels=['Paid Off', 'Not Paid Off'],autopct='%1.1f%%')
        plt.show()

    def six_months(self):
        """
        Function to find the expected paid off amount in six months time
        """
        counter = 0
        six_month_percent = []
        for value in self.df['instalment']:
            if ((value*6)+self.df['total_rec_prncp'].iloc[counter]) > self.df['loan_amount'].iloc[counter]:
                six_month_percent.append(self.df['loan_amount'].iloc[counter])
            else:
                six_month_percent.append((value*6)+self.df['total_rec_prncp'].iloc[counter])
            counter += 1
        six_month_percent = pd.Series(six_month_percent)
        loan_percent = (sum(six_month_percent)/sum(self.df['loan_amount']))*100
        plt.pie([loan_percent,100-loan_percent], labels=['Paid Off', 'Not Paid Off'],autopct='%1.1f%%')
        plt.show()
    
    def loan_loss(self):
        """
        Function to find the percent of loans that are paid off and the total loss from charged off loans
        """
        counter = 0
        charged_off_counter = 0
        loan_total = 0
        paid_total = 0
        mnths_paid = 0
        total_loss = 0
        for value in self.df['loan_status']:
            if value == 'Charged Off':
                charged_off_counter += 1
                loan_total += self.df['loan_amount'].iloc[counter]
                paid_total += self.df['total_rec_prncp'].iloc[counter]
                mnths_paid = round(self.df['total_payment'].iloc[counter]/self.df['instalment'].iloc[counter])
                mnths_left = self.df['term'].iloc[counter]-mnths_paid
                total_loss += mnths_left*self.df['instalment'].iloc[counter]
            counter += 1
        percent_charged_off = (charged_off_counter/len(self.df['loan_status']))*100
        print('The percent that has been charged off is '+str(round(percent_charged_off,2))+'%'+
              ', and the amount paid back is '+str(round(paid_total,2))+' out of '+str(round(loan_total,2)))
        print('The amount that the company lost due to these charged off loans is, '+str(round(total_loss,2)))

    def possible_loss(self):
        """
        Function to find the possible loss if late loans were to be charged off
        """
        late_count = 0
        counter = 0
        total_loss = 0
        proj_loss = 0
        exp_rev = 0
        for value in self.df['loan_status']:
            if 'Late' in value:
                late_count += 1
                proj_loss += self.df['out_prncp'].iloc[counter]
            elif value == 'Charged Off':
                total_loss += (self.df['loan_amount'].iloc[counter]- self.df['total_payment'].iloc[counter])
            counter += 1
        counter = 0
        for value in self.df['instalment']:
            exp_rev += self.df['term'].iloc[counter]*value
            counter += 1
        late_percent = late_count/len(self.df['loan_status'])*100
        total_loss += proj_loss
        percent_loss = (total_loss/exp_rev)*100
        print('The percentage of users that have a late payment is '+str(round(late_percent,2))+
        '%, the projected loss if said customers were to not finish their payment is '+str(round(proj_loss,2))+
        '\nand the percent of loss for all late loans if they were charged off, as well as all currently charged off loans is '
        +str(round(percent_loss,2))+'%')

    def indicator_of_loss(self):
        """
        Function that focusses on columns that are most likely to affect the chance of a loan being charged off
        """
        counter = 0
        # Focus on the grade column
        grade_count_loss = {
            'A':0,
            'B':0,
            'C':0,
            'D':0,
            'E':0,
            'F':0,
            'G':0
        }
        grade_count = {
            'A':0,
            'B':0,
            'C':0,
            'D':0,
            'E':0,
            'F':0,
            'G':0
        }
        for value in self.df['grade']:
            grade_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                grade_count_loss[value] += 1
            counter += 1
        grade_percent = {key: round((grade_count_loss[key] / grade_count.get(key, 0))*100,2)
                        for key in grade_count_loss.keys()}
        counter = 0
        # Focus on the purpose column
        purpose_count = {
            'credit_card':0,
            'debt_consolidation':0,
            'home_improvement':0,
            'small_business':0,
            'renewable_energy':0,
            'major_purchase':0,
            'other':0,
            'moving':0,
            'car':0,
            'medical':0,
            'house':0,
            'vacation':0,
            'wedding':0,
            'educational':0
        }
        purpose_count_loss = {
            'credit_card':0,
            'debt_consolidation':0,
            'home_improvement':0,
            'small_business':0,
            'renewable_energy':0,
            'major_purchase':0,
            'other':0,
            'moving':0,
            'car':0,
            'medical':0,
            'house':0,
            'vacation':0,
            'wedding':0,
            'educational':0
        }
        for value in self.df['purpose']:
            purpose_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                purpose_count_loss[value] += 1
            counter += 1
        purpose_percent = {key: round((purpose_count_loss[key] / purpose_count.get(key, 0))*100,2)
                        for key in purpose_count_loss.keys()}
        counter = 0
        # Focus on the home_ownership column
        home_count = {
            'MORTGAGE':0,
            'RENT':0,
            'OWN':0,
            'OTHER':0,
            'NONE':0
        }
        home_count_loss = {
            'MORTGAGE':0,
            'RENT':0,
            'OWN':0,
            'OTHER':0,
            'NONE':0
        }
        for value in self.df['home_ownership']:
            home_count[value] += 1
            if self.df['loan_status'].iloc[counter] == 'Charged Off' or 'Late' in self.df['loan_status'].iloc[counter]:
                home_count_loss[value] += 1
            counter += 1
        home_percent = {key: round((home_count_loss[key] / home_count.get(key, 0))*100,2)
                        for key in home_count_loss.keys()}#
        # Final result
        print('We can see that the lower the grade level of a loan the more likely it is to affect if the '+
              'loan is charged off or late:\n',grade_percent)
        print('The purpose of the loan seems to not have too big of an impact on the likelihood of '+
               'the loan being a loss:\n',purpose_percent,', execpt if the purpose is small business.')
        print('The home ownership has so imapct on the chance of the loan being '+
               'charged off or late:\n',home_percent)