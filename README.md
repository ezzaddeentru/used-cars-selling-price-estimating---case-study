# used-cars-selling-price-estimating---case-study

## Business objective
Estimating the sales price of cars within 10% of the listed price.
<br><br>

### **What are the current solutions/workarounds (if any)?**

Currently, when a new car comes in, team members take all of the information that usually appears in the advert and give it to this team member. They then estimate the price. We have been testing the team members estimating themselves but they are always around **30%** away from the price we know the car will sell for. 
<br><br>

### **What would be the minimum performance needed to reach the business objective?**

The team estimates are always around 30% off, we really want to be **within 10%** of the price. This will mean we can automate the whole process and be able to sell cars quicker.

They want us to predict prices within 10% of the listed price. But as their team can only manage 30%, it is probably ok to show we are at least as good as that.
<br><br>

### **How should performance be measured?**

The metric most related to our goal of being within a 10% error margin for estimating car selling prices is **the Mean Absolute Percentage Error (MAPE)**. MAPE directly measures the average percentage difference between predicted and actual prices, which aligns well with our requirement to ensure that the predictions are within a specific percentage range of the actual selling prices.


## Data information
![image](https://github.com/user-attachments/assets/6edfb661-52c4-4ef5-97a1-994ea2699c21)
