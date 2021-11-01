from config import *
train_data = pd.read_csv("data/train_data/train_feature.csv")
test_data = pd.read_csv("data/test_data/test_feature.csv")



campaign_data = pd.read_csv("data/campaign_data.csv")
coupon_item = pd.read_csv("data/coupon_item_mapping.csv")
customer_data = pd.read_csv("data/customer_demographics.csv")
transaction_data = pd.read_csv("data/customer_transaction_data.csv")
item_data = pd.read_csv("data/item_data.csv")
test = pd.read_csv("data/test_data.csv")
train = pd.read_csv("data/train_data.csv")



train_data.drop(['c_unique_items', 'c_unique_brand', 'c_freq_brand',
       'c_rare_brand', 'c_items_freq_brand', 'c_items_rare_brand','c_unique_brandt', 'c_freq_brandt', 'c_rare_brandt','c_items_freq_brandt', 
       'c_items_rare_brandt', 'c_unique_category',
       'c_freq_category', 'c_rare_category', 'c_items_freq_category',
       'c_items_rare_category', 'c_coverage_item', 'c_coverage_brand',
       'c_coverage_brandt', 'c_coverage_category', 'overall_unique_items',
       'overall_items', 'overall_quantity', 'overall_sprice', 'overall_bprice',
       'overall_odiscount', 'overall_cdiscount', 'overall_tdiscount',
       'overall_sprice_pq', 'overall_bprice_pq', 'overall_odiscount_pq',
       'overall_cdiscount_pq', 'overall_tdiscount_pq', 'overall_unique_brand',
       'overall_freq_brand', 'overall_rare_brand', 'overall_items_freq_brand',
       'overall_items_rare_brand', 'overall_unique_brandt',
       'overall_freq_brandt', 'overall_rare_brandt',
       'overall_items_freq_brandt', 'overall_items_rare_brandt',
       'overall_unique_category', 'overall_freq_category',
       'overall_rare_category', 'overall_items_freq_category',
       'overall_items_rare_category', 'overall_coverage_item',
       'overall_coverage_brand', 'overall_coverage_brandt',
       'overall_coverage_category', 'overall_podiscount', 'overall_pcdiscount',
       'overall_ptdiscount', 'overall_podiscount_pq', 'overall_pcdiscount_pq',
       'overall_ptdiscount_pq'], axis = 1, inplace=True)


print("Important features ")
Important_features = (train_data[train_data.columns[:]].corr()['redemption_status'][:])
Important_features = (Important_features.sort_values(ascending=False))

print(list(Important_features.drop('redemption_status').head().index.values))




customer_data = train_data
customer_data = customer_data.drop(['id','coupon_id','campaign_id','campaign_type','duration','start_date','end_date','age_range'] ,axis=1)
print("Important Customer Attributes")
customer_features = (customer_data[customer_data.columns[:]].corr()['redemption_status'][:])
customer_features = (customer_features.sort_values(ascending=False))

print(list(customer_features.drop('redemption_status').head().index.values))


# Predicting Unseful coupons 
coupon_data = train_data[['coupon_id','redemption_status']]
useful_coupon = pd.merge(coupon_data,coupon_item[['coupon_id','item_id']],on='coupon_id', how='left')
useful_coupon = pd.merge(useful_coupon,item_data[['item_id','category']],on='item_id', how='left')
#print(useful_coupon.columns)

import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec

'''
# Create 2x2 sub plots
gs = gridspec.GridSpec(2, 2)
pl.figure()
ax = pl.subplot(gs[0, 0]) # row 0, col 0
pl.scatter(useful_coupon['coupon_id'], useful_coupon['redemption_status'])

ax = pl.subplot(gs[0, 1]) # row 0, col 1
pl.scatter(useful_coupon['item_id'], useful_coupon['redemption_status'] )

ax = pl.subplot(gs[1, :]) # row 1, span all columns
pl.scatter(useful_coupon['category'], useful_coupon['redemption_status'])
pl.show()
'''


category_redemption = useful_coupon.groupby(['category','redemption_status']).redemption_status.count()

category_redemption_percentage = category_redemption.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))


print(" Redemption percentage with Coupons and Categories ")
print(category_redemption_percentage.sort_values(ascending=False))



# Profiling Items

df = pd.merge(transaction_data,item_data, on="item_id",how = 'left')
print(df)
total = df['other_discount'] + df['coupon_discount']
df['total_discount'] = total
percent_discount = (-1) * 100 * df['total_discount'] / df['selling_price']
df['percent_discount'] = percent_discount
df = df.drop(['date','customer_id','brand','brand_type','other_discount','coupon_discount','total_discount'],axis = 1)
print("Most sold products by category")
#print(df.groupby('category').size().sort_values(ascending=False).head())
print(df.groupby('category').agg('mean').sort_values(by='percent_discount', ascending=False).head().drop(['item_id','quantity','selling_price'],axis=1))
print(df.groupby('category').agg('mean').sort_values(by='selling_price', ascending=False).head().drop(['item_id','quantity','percent_discount'],axis=1))



df = campaign_data
df["date"] = df.apply(
    lambda x: pd.date_range(x["start_date"], x["end_date"]), axis=1
)
df = (
    df.explode("date", ignore_index=True)
    .drop(columns=["start_date", "end_date"])
)
print(df)

'''
#df = pd.concat([df,transaction_data],join='inner',axis=1)
#df = df.merge(transaction_data, left_index=True, right_index=True, how='inner')
df = pd.concat([df.set_index('date'),transaction_data.set_index('date')], axis=1).fillna(method='ffill')

print(df)
'''


