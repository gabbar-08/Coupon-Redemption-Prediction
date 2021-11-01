from config import *


campaign_data = pd.read_csv("data/campaign_data.csv")
coupon_item = pd.read_csv("data/coupon_item_mapping.csv")
customer_data = pd.read_csv("data/customer_demographics.csv")
transaction_data = pd.read_csv("data/customer_transaction_data.csv")
item_data = pd.read_csv("data/item_data.csv")
test_data = pd.read_csv("data/test_data.csv")
train_data = pd.read_csv("data/train_data.csv")

item_data['brand_type'] = item_data['brand_type'].replace({'Established': 1, 'Local': 0})

total_items = item_data['item_id'].nunique()
total_brands= item_data['brand'].nunique()
total_brand_types = item_data['brand_type'].nunique()
total_categories = item_data['category'].nunique()

campaign_data = pd.read_csv('data/campaign_data.csv', parse_dates=['start_date', 'end_date'], dayfirst=True)

campaign_data['campaign_type'] = campaign_data['campaign_type'].replace({'X': 0, 'Y': 1})
campaign_data['duration'] = (campaign_data['end_date'] - campaign_data['start_date']).dt.days


campaign_data_index = campaign_data.set_index('campaign_id')


def get_marital_status(row):
    na_row = row.isna()
    if not na_row['marital_status']:
        return row['marital_status']
    return 'Married' if row['family_size'] - row['no_of_children'] > 1 else 'Single'

customer_data['family_size'] = customer_data['family_size'].str.replace('+','',regex=True).astype('int')
customer_data['no_of_children'] = customer_data['no_of_children'].fillna('0').str.replace('+','',regex=True).astype('int')
customer_data['marital_status'] = customer_data.apply(get_marital_status, axis=1)
customer_data['marital_status'] = customer_data['marital_status'].replace({'Single': 0, 'Married': 1})

coupon_item = coupon_item.merge(item_data, how='left', on='item_id')
coupon_item_index = coupon_item.set_index('coupon_id')



# DATA Preprocessing
columns = train_data.columns[train_data.columns != 'redemption_status']
total_data = train_data[columns].append(test_data, sort=True)
# Step 1: - Creating coupon related variables such as most frequent and least frequent

def most_frequent(s):
    return s.value_counts().index[0]

def least_frequent(s):
    return s.value_counts().index[-1]

def most_frequent_count(s):
    return s.value_counts().values[0]

def least_frequent_count(s):
    return s.value_counts().values[-1]

coupon_data = coupon_item.groupby('coupon_id').agg({
    'item_id': ['nunique'],
    'brand': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count],
    'brand_type': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count],
    'category': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count]
})
coupon_data.columns = ['c_unique_items', 'c_unique_brand', 'c_freq_brand', 'c_rare_brand', 
                       'c_items_freq_brand', 'c_items_rare_brand', 'c_unique_brandt', 'c_freq_brandt',
                       'c_rare_brandt', 'c_items_freq_brandt', 'c_items_rare_brandt', 
                       'c_unique_category', 'c_freq_category', 'c_rare_category', 'c_items_freq_category', 
                       'c_items_rare_category']
coupon_data['c_coverage_item'] = coupon_data['c_unique_items'] / total_items
coupon_data['c_coverage_brand'] = coupon_data['c_unique_brand'] / total_brands
coupon_data['c_coverage_brandt'] = coupon_data['c_unique_brandt'] / total_brand_types
coupon_data['c_coverage_category'] = coupon_data['c_unique_category'] / total_categories

#print(coupon_data.info())

# Step 2 :- Creating Customer realted varibales into per quantity data
transaction_data['total_discount'] = transaction_data['coupon_discount'] + transaction_data['other_discount']
transaction_data['buying_price'] = transaction_data['selling_price'] + transaction_data['other_discount']
transaction_data['selling_price_pq'] = transaction_data['selling_price'] / transaction_data['quantity']
transaction_data['other_discount_pq'] = transaction_data['other_discount'] / transaction_data['quantity']
transaction_data['coupon_discount_pq'] = transaction_data['coupon_discount'] / transaction_data['quantity']
transaction_data['total_discount_pq'] = transaction_data['coupon_discount_pq'] + transaction_data['other_discount_pq']
transaction_data['buying_price_pq'] = transaction_data['selling_price_pq'] + transaction_data['other_discount_pq']
transaction_data['date'] = pd.to_datetime(transaction_data['date'])
transaction_data = transaction_data.merge(item_data, on='item_id', how='left')
transaction_data = transaction_data.set_index(['customer_id','date']).sort_index()

# Step 3 :- combining results from Step 1 and 2

customer_history = transaction_data.groupby('customer_id').agg({
    'item_id': ['nunique', 'count'],
    'quantity': 'sum',
    'selling_price': 'mean',
    'buying_price': 'mean',
    'other_discount': 'mean',
    'coupon_discount': 'mean',
    'total_discount': 'mean',
    'selling_price_pq': 'mean',
    'buying_price_pq': 'mean',
    'other_discount_pq': 'mean',
    'coupon_discount_pq': 'mean',
    'total_discount_pq': 'mean',
    'brand': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count],
    'brand_type': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count],
    'category': ['nunique', most_frequent, least_frequent, most_frequent_count, least_frequent_count]
})
customer_history.columns = ['overall_unique_items', 'overall_items', 'overall_quantity', 'overall_sprice', 'overall_bprice', 'overall_odiscount', 'overall_cdiscount', 'overall_tdiscount', 'overall_sprice_pq', 'overall_bprice_pq', 'overall_odiscount_pq', 'overall_cdiscount_pq', 'overall_tdiscount_pq', 'overall_unique_brand', 'overall_freq_brand', 'overall_rare_brand', 'overall_items_freq_brand', 'overall_items_rare_brand', 'overall_unique_brandt', 'overall_freq_brandt', 'overall_rare_brandt', 'overall_items_freq_brandt', 'overall_items_rare_brandt', 'overall_unique_category', 'overall_freq_category', 'overall_rare_category', 'overall_items_freq_category', 'overall_items_rare_category']
customer_history['overall_coverage_item'] = customer_history['overall_unique_items'] / total_items
customer_history['overall_coverage_brand'] = customer_history['overall_unique_brand'] / total_brands
customer_history['overall_coverage_brandt'] = customer_history['overall_unique_brandt'] / total_brand_types
customer_history['overall_coverage_category'] = customer_history['overall_unique_category'] / total_categories
customer_history['overall_podiscount'] = customer_history['overall_odiscount'] / customer_history['overall_bprice']
customer_history['overall_pcdiscount'] = customer_history['overall_cdiscount'] / customer_history['overall_bprice']
customer_history['overall_ptdiscount'] = customer_history['overall_tdiscount'] / customer_history['overall_bprice']
customer_history['overall_podiscount_pq'] = customer_history['overall_odiscount_pq'] / customer_history['overall_bprice_pq']
customer_history['overall_pcdiscount_pq'] = customer_history['overall_cdiscount_pq'] / customer_history['overall_bprice_pq']
customer_history['overall_ptdiscount_pq'] = customer_history['overall_tdiscount_pq'] / customer_history['overall_bprice_pq']
customer_history = customer_history.reset_index()

#Step 4 :- Combining outcomes of Step 3 with original data from database

total_data = total_data.merge(campaign_data, on='campaign_id', how='left')
total_data = total_data.merge(customer_data, on='customer_id', how='left')
total_data = total_data.merge(coupon_data, on='coupon_id', how='left')
total_data = total_data.merge(customer_history, on='customer_id', how='left')

#Step 5 :- Creating Training and Testing data .csv fies with complete data 

test_data = test_data[['id']].merge(total_data, on='id', how='left')
#print(test_data.info())
train_data = train_data[['id','redemption_status']].merge(total_data, on='id', how='left')
#print(train_data.info())
test_data.to_csv('data/test_data/test_feature.csv', index=False)
train_data.to_csv('data/train_data/train_feature.csv', index=False)
