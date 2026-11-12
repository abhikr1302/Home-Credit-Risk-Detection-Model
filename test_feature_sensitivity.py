import copy
import requests
from test_model_cases import BASE_CASE
API_URL = 'http://127.0.0.1:8000/predict'

def predict(payload):
    response = requests.post(API_URL, json=payload, timeout=60)
    if response.status_code != 200:
        print('API ERROR:')
        print(response.text)
        raise SystemExit(1)
    result = response.json()
    return result['risk_probability']

def test_feature(feature_name, values):
    print('\n' + '=' * 90)
    print(f'FEATURE: {feature_name}')
    print('=' * 90)
    print(f"{'VALUE':>15}{'PROBABILITY':>18}{'RISK %':>15}")
    print('-' * 90)
    results = []
    for value in values:
        case = copy.deepcopy(BASE_CASE)
        case[feature_name] = value
        probability = predict(case)
        results.append((value, probability))
        print(f'{str(value):>15}{probability:>17.6f}{probability * 100:>14.2f}%')
    probabilities = [probability for _, probability in results]
    minimum = min(probabilities)
    maximum = max(probabilities)
    change = maximum - minimum
    print('-' * 90)
    print(f'Minimum probability : {minimum:.6f}')
    print(f'Maximum probability : {maximum:.6f}')
    print(f'Range               : {change:.6f}')
    if change == 0:
        print('⚠️ NO MODEL RESPONSE CHANGE')
    elif change < 0.01:
        print('⚠️ VERY SMALL MODEL RESPONSE')
    else:
        print('✅ MODEL RESPONDS TO FEATURE')

def main():
    test_feature('credit_score', [450, 550, 650, 700, 750, 800, 850])
    test_feature('external_score_1', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    test_feature('external_score_2', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    test_feature('external_score_3', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    test_feature('annual_income', [200000, 300000, 500000, 750000, 1000000, 1500000, 2500000])
    test_feature('loan_amount', [50000, 100000, 200000, 300000, 500000, 750000, 1000000])
    test_feature('monthly_emi', [2000, 4000, 6000, 7500, 10000, 15000, 25000, 40000])
    test_feature('total_outstanding_debt', [0, 50000, 100000, 250000, 500000, 750000, 1000000])
    test_feature('total_overdue_amount', [0, 1000, 5000, 10000, 25000, 50000, 100000])
    print('\n\n')
    print('=' * 90)
    print('CREDIT CARD UTILIZATION')
    print('=' * 90)
    for utilization in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
        case = copy.deepcopy(BASE_CASE)
        case['credit_card_limit'] = 200000
        case['credit_card_balance'] = 200000 * utilization / 100
        case['credit_card_max_utilization'] = utilization
        probability = predict(case)
        print(f'Utilization {utilization:>3}% -> {probability * 100:>7.2f}%')
    test_feature('late_installment_rate', [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8])
    test_feature('maximum_installment_delay', [0, 5, 15, 30, 60, 90, 120, 180])
    test_feature('average_payment_ratio', [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    print('\n\n')
    print('=' * 90)
    print('PREVIOUS APPLICATION REFUSALS')
    print('=' * 90)
    for refused in range(0, 9):
        case = copy.deepcopy(BASE_CASE)
        case['previous_applications'] = 8
        case['previous_refused'] = refused
        case['previous_approved'] = 8 - refused
        probability = predict(case)
        print(f'Refused {refused:>2} -> {probability * 100:>7.2f}%')
    test_feature('social_circle_30plus_defaults', [0, 1, 2, 3, 5, 10])
    test_feature('social_circle_60plus_defaults', [0, 1, 2, 3, 5, 10])
    test_feature('pos_cash_max_dpd', [0, 10, 30, 60, 90, 120, 180])
    test_feature('credit_card_30plus_dpd', [0, 1, 2, 3, 5, 10])
    print('\n')
    print('=' * 90)
    print('FEATURE SENSITIVITY TESTING COMPLETE')
    print('=' * 90)
if __name__ == '__main__':
    main()
