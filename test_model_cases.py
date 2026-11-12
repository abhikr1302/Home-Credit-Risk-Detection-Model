import copy
import requests
API_URL = 'http://127.0.0.1:8000'
BASE_CASE = {'age': 35, 'gender': 'Male', 'marital_status': 'Married', 'family_members': 2, 'education': 'Higher education', 'housing_type': 'House / apartment', 'residential_status': 'Owned', 'owns_car': True, 'owns_property': True, 'contract_type': 'Cash loans', 'income_type': 'Working', 'employment_years': 5.0, 'annual_income': 600000, 'other_monthly_income': 0, 'income_stability_years': 5, 'loan_type': 'Personal Loan', 'loan_purpose': 'Other', 'secured_loan': False, 'loan_amount': 300000, 'loan_term_months': 60, 'monthly_emi': 7500, 'credit_score': 700, 'external_score_1': 0.7, 'external_score_2': 0.7, 'external_score_3': 0.7, 'total_credit_accounts': 5, 'active_credit_accounts': 2, 'closed_credit_accounts': 3, 'total_credit_limit': 1000000, 'total_outstanding_debt': 250000, 'total_overdue_amount': 0, 'average_credit_history_months': 48, 'maximum_days_past_due': 0, 'delinquent_accounts': 0, 'number_of_credit_cards': 2, 'credit_card_balance': 50000, 'credit_card_limit': 200000, 'credit_card_30plus_dpd': 0, 'credit_card_max_utilization': 30, 'credit_card_average_balance': 25000, 'previous_applications': 3, 'previous_approved': 2, 'previous_refused': 1, 'previous_average_loan_amount': 200000, 'previous_average_emi': 6000, 'previous_average_credit_application_ratio': 0.7, 'previous_loans_with_installments': 3, 'installment_records': 36, 'late_installment_rate': 0.05, 'maximum_installment_delay': 5, 'average_payment_ratio': 0.98, 'pos_cash_loans': 3, 'pos_cash_active_loans': 1, 'pos_cash_history_months': 36, 'pos_cash_max_dpd': 0, 'social_circle_30plus_defaults': 0, 'social_circle_60plus_defaults': 0, 'document_risk_flag': 0, 'recent_credit_enquiries': 1, 'fraud_flag': False, 'kyc_verified': True}

def main():
    TEST_CASES = []
    TEST_CASES.append(('01 - Baseline', BASE_CASE))
    case = copy.deepcopy(BASE_CASE)
    case.update({'age': 40, 'employment_years': 10, 'annual_income': 1500000, 'loan_amount': 300000, 'monthly_emi': 7000, 'credit_score': 800, 'external_score_1': 0.9, 'external_score_2': 0.9, 'external_score_3': 0.9, 'total_credit_accounts': 8, 'active_credit_accounts': 2, 'closed_credit_accounts': 6, 'total_credit_limit': 2500000, 'total_outstanding_debt': 150000, 'total_overdue_amount': 0, 'maximum_days_past_due': 0, 'delinquent_accounts': 0, 'credit_card_balance': 30000, 'credit_card_limit': 500000, 'credit_card_max_utilization': 6, 'credit_card_30plus_dpd': 0, 'previous_applications': 5, 'previous_approved': 5, 'previous_refused': 0, 'late_installment_rate': 0.0, 'maximum_installment_delay': 0, 'average_payment_ratio': 1.0, 'pos_cash_loans': 4, 'pos_cash_active_loans': 1, 'pos_cash_max_dpd': 0, 'social_circle_30plus_defaults': 0, 'social_circle_60plus_defaults': 0, 'document_risk_flag': 0, 'recent_credit_enquiries': 0})
    TEST_CASES.append(('02 - Very Strong Applicant', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'age': 23, 'employment_years': 0.5, 'annual_income': 300000, 'loan_amount': 500000, 'monthly_emi': 18000, 'credit_score': 520, 'external_score_1': 0.2, 'external_score_2': 0.2, 'external_score_3': 0.2, 'total_credit_accounts': 8, 'active_credit_accounts': 7, 'closed_credit_accounts': 1, 'total_credit_limit': 400000, 'total_outstanding_debt': 280000, 'total_overdue_amount': 50000, 'maximum_days_past_due': 120, 'delinquent_accounts': 4, 'credit_card_balance': 180000, 'credit_card_limit': 200000, 'credit_card_max_utilization': 90, 'credit_card_30plus_dpd': 3, 'previous_applications': 5, 'previous_approved': 1, 'previous_refused': 4, 'late_installment_rate': 0.45, 'maximum_installment_delay': 90, 'average_payment_ratio': 0.7, 'pos_cash_loans': 6, 'pos_cash_active_loans': 5, 'pos_cash_history_months': 60, 'pos_cash_max_dpd': 90, 'social_circle_30plus_defaults': 3, 'social_circle_60plus_defaults': 2, 'document_risk_flag': 1, 'recent_credit_enquiries': 8})
    TEST_CASES.append(('03 - Very Weak Applicant', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'annual_income': 600000, 'loan_amount': 500000, 'monthly_emi': 25000, 'total_outstanding_debt': 500000, 'total_overdue_amount': 0, 'credit_card_balance': 150000, 'credit_card_limit': 200000, 'credit_card_max_utilization': 75})
    TEST_CASES.append(('04 - High Debt / High EMI', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'annual_income': 300000, 'loan_amount': 150000, 'monthly_emi': 4500, 'total_credit_accounts': 2, 'active_credit_accounts': 1, 'closed_credit_accounts': 1, 'total_credit_limit': 300000, 'total_outstanding_debt': 50000, 'total_overdue_amount': 0, 'maximum_days_past_due': 0, 'delinquent_accounts': 0, 'credit_card_balance': 10000, 'credit_card_limit': 100000, 'credit_card_max_utilization': 10, 'previous_applications': 1, 'previous_approved': 1, 'previous_refused': 0, 'late_installment_rate': 0, 'maximum_installment_delay': 0, 'average_payment_ratio': 1, 'recent_credit_enquiries': 0})
    TEST_CASES.append(('05 - Low Income but Clean History', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'credit_card_balance': 190000, 'credit_card_limit': 200000, 'credit_card_max_utilization': 95, 'credit_card_30plus_dpd': 0})
    TEST_CASES.append(('06 - 95% Credit Card Utilization', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'late_installment_rate': 0.6, 'maximum_installment_delay': 120, 'average_payment_ratio': 0.6, 'credit_card_30plus_dpd': 5, 'maximum_days_past_due': 120, 'delinquent_accounts': 4, 'total_overdue_amount': 75000, 'pos_cash_max_dpd': 90})
    TEST_CASES.append(('07 - Serious Repayment Problems', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'previous_applications': 8, 'previous_approved': 1, 'previous_refused': 7, 'previous_average_credit_application_ratio': 1.2})
    TEST_CASES.append(('08 - Many Previous Refusals', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'credit_score': 850, 'external_score_1': 0.95, 'external_score_2': 0.95, 'external_score_3': 0.95, 'total_credit_accounts': 10, 'active_credit_accounts': 3, 'closed_credit_accounts': 7, 'total_credit_limit': 3000000, 'total_outstanding_debt': 100000, 'total_overdue_amount': 0, 'credit_card_balance': 20000, 'credit_card_limit': 500000, 'credit_card_max_utilization': 4, 'previous_applications': 6, 'previous_approved': 6, 'previous_refused': 0, 'late_installment_rate': 0, 'maximum_installment_delay': 0, 'average_payment_ratio': 1, 'recent_credit_enquiries': 0})
    TEST_CASES.append(('09 - Excellent Credit Profile', case))
    case = copy.deepcopy(BASE_CASE)
    case.update({'age': 21, 'employment_years': 0.2, 'annual_income': 240000, 'loan_amount': 600000, 'monthly_emi': 30000, 'credit_score': 450, 'external_score_1': 0.1, 'external_score_2': 0.1, 'external_score_3': 0.1, 'total_credit_accounts': 10, 'active_credit_accounts': 9, 'closed_credit_accounts': 1, 'total_credit_limit': 300000, 'total_outstanding_debt': 280000, 'total_overdue_amount': 100000, 'maximum_days_past_due': 180, 'delinquent_accounts': 7, 'credit_card_balance': 290000, 'credit_card_limit': 300000, 'credit_card_max_utilization': 97, 'credit_card_30plus_dpd': 6, 'previous_applications': 10, 'previous_approved': 0, 'previous_refused': 10, 'late_installment_rate': 0.8, 'maximum_installment_delay': 180, 'average_payment_ratio': 0.5, 'pos_cash_loans': 10, 'pos_cash_active_loans': 9, 'pos_cash_history_months': 60, 'pos_cash_max_dpd': 180, 'social_circle_30plus_defaults': 5, 'social_circle_60plus_defaults': 4, 'document_risk_flag': 1, 'recent_credit_enquiries': 15, 'fraud_flag': True, 'kyc_verified': False})
    TEST_CASES.append(('10 - Extreme Risk', case))
    print('=' * 90)
    print('HOME CREDIT MODEL - MULTI CASE TEST')
    print('=' * 90)
    try:
        response = requests.get(f'{API_URL}/health', timeout=10)
    except Exception as exc:
        print(f'\n❌ Cannot connect to API: {exc}')
        print('\nStart the API with:')
        print('uvicorn src.api:app --reload')
        raise SystemExit(1)
    if response.status_code != 200:
        print('❌ API health check failed')
        print(response.text)
        raise SystemExit(1)
    health = response.json()
    print('\nMODEL CONFIGURATION')
    print('-' * 90)
    print(f"Model loaded       : {health.get('model_loaded')}")
    print(f"Model              : {health.get('model')}")
    print(f"Feature count      : {health.get('feature_count')}")
    print(f"Threshold          : {health.get('threshold')}")
    print(f"Backend engineering: {health.get('backend_feature_engineering')}")
    results = []
    for name, payload in TEST_CASES:
        print('\n' + '=' * 90)
        print(name)
        print('=' * 90)
        try:
            response = requests.post(f'{API_URL}/predict', json=payload, timeout=60)
        except Exception as exc:
            print(f'❌ API error: {exc}')
            continue
        if response.status_code != 200:
            print(f'❌ HTTP {response.status_code}')
            print(response.text)
            continue
        result = response.json()
        probability = result.get('risk_probability')
        percentage = result.get('risk_percentage')
        risk_level = result.get('risk_level')
        decision = result.get('decision')
        affordability = result.get('affordability', {})
        flags = result.get('risk_flags', [])
        backend = result.get('backend_processing', {})
        print(f'Probability       : {probability:.6f}')
        print(f'Risk %            : {percentage:.2f}%')
        print(f'Risk level        : {risk_level}')
        print(f'Decision           : {decision}')
        print(f"DTI               : {affordability.get('debt_to_income_percentage', 0):.2f}%")
        print(f"Model features    : {backend.get('features_sent_to_model')}")
        if flags:
            print('Risk flags:')
            for flag in flags:
                print(f'  - {flag}')
        results.append({'name': name, 'probability': probability, 'risk_level': risk_level, 'decision': decision})
    print('\n\n')
    print('=' * 90)
    print('SUMMARY')
    print('=' * 90)
    print(f"{'TEST CASE':<38}{'PROBABILITY':>15}{'RISK':>12}{'DECISION':>20}")
    print('-' * 90)
    for result in results:
        print(f"{result['name']:<38}{result['probability'] * 100:>14.2f}%{result['risk_level']:>12}{result['decision']:>20}")
    print('\n')
    print('=' * 90)
    print('SANITY CHECK')
    print('=' * 90)
    if len(results) >= 3:
        baseline = results[0]['probability']
        strong = next((r['probability'] for r in results if 'Very Strong' in r['name']), None)
        weak = next((r['probability'] for r in results if 'Very Weak' in r['name']), None)
        extreme = next((r['probability'] for r in results if 'Extreme Risk' in r['name']), None)
        if strong is not None:
            print(f'\nBaseline probability : {baseline:.4f}')
            print(f'Strong probability   : {strong:.4f}')
            if strong < baseline:
                print('✅ Strong applicant probability is lower than baseline.')
            else:
                print('⚠️ Strong applicant probability is NOT lower than baseline.')
        if weak is not None:
            print(f'\nWeak probability     : {weak:.4f}')
            if weak > baseline:
                print('✅ Weak applicant probability is higher than baseline.')
            else:
                print('⚠️ Weak applicant probability is NOT higher than baseline.')
        if extreme is not None:
            print(f'\nExtreme probability  : {extreme:.4f}')
            if extreme > baseline:
                print('✅ Extreme-risk probability is higher than baseline.')
            else:
                print('⚠️ Extreme-risk probability is NOT higher than baseline.')
    print('\n')
    print('Testing complete.')
if __name__ == '__main__':
    main()
