--##############################################################################################################
-- Stored Procedure to create the new sales (>90 days) Training Data asset (PSR4 and PSR5). The difference between
-- PSR4 (propensity for ACH+) and PSR5 (propensity for VCC) is the target variable.
-- Arguments
-- data_set (STRING): Data Set (e.g. training)

--
-- Next Steps: Consider adding validations for:
-- making sure data_set is only training. No other strings should be allowed (avoid typos).
--
--
-- How to Call SP:
--CALL DEV_ZADA_DATASCIENCE.pymts_sales_propensity_all_time.PSR4_PSR5_Data_Asset_RF_SP('training');
--##############################################################################################################


CREATE OR REPLACE PROCEDURE DEV_ZADA_DATASCIENCE.pymts_sales_propensity_all_time.PSR4_PSR5_Data_Asset_RF_SP(
    data_set STRING -- Data Set (training)
    )
RETURNS STRING
LANGUAGE SQL
EXECUTE AS OWNER
AS '\n

DECLARE\n    
final_table_name STRING := \'dev_zada_datascience.pymts_sales_propensity_all_time.PSR4_PSR5_RF_\' || data_set;\n
result_message STRING := \'PSR4 and PSR5 Data Asset successfully created for provider accounts created on or after 2012 \';\n

BEGIN\n

--##############################################################################################################
-- Initial payments table
--##############################################################################################################

create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF as (
with payments as (
select * from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_PAYMENTS
where statusname in (\'Processed - Paid\', \'Processing - Out for payment\') and paymentamount > 1)
,providers as (
select * from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_STATUS
where providercreatedon >= \'2012-01-01\'
)
select p1.*, p2.providercreatedon
,CASE WHEN p1.productline = \'Check\' THEN p1.paymentcount ELSE 0 END AS checkpaymentcount
,CASE WHEN p1.productline = \'Check\' THEN p1.paymentamount ELSE 0 END AS checkpaymentamount
,CASE WHEN p1.productline = \'VCC\' THEN p1.paymentcount ELSE 0 END AS vccpaymentcount
,CASE WHEN p1.productline = \'VCC\' THEN p1.paymentamount ELSE 0 END AS vccpaymentamount
,CASE WHEN p1.productline = \'ACH+\' THEN p1.paymentcount ELSE 0 END AS achpaymentcount
,CASE WHEN p1.productline = \'ACH+\' THEN p1.paymentamount ELSE 0 END AS achpaymentamount
,CASE WHEN p1.productline = \'PayerSponsoredACH\' THEN p1.paymentcount ELSE 0 END AS payersponsoredpaymentcount
,CASE WHEN p1.productline = \'PayerSponsoredACH\' THEN p1.paymentamount ELSE 0 END AS payersponsoredpaymentamount
,CASE WHEN p1.productline = \'ACH Reversal\' THEN p1.paymentcount ELSE 0 END AS reversalpaymentcount
,CASE WHEN p1.productline = \'ACH Reversal\' THEN p1.paymentamount ELSE 0 END AS reversalpaymentamount
from payments p1
inner join providers p2
on p1.TIN = p2.TIN and p1.providerid = p2.providerid
);

--##############################################################################################################
-- This table has all the Cancelled payments for providers whose accounts were created 2012 or newer
--##############################################################################################################
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF as (
with payments as (
select * from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_PAYMENTS
where statusname = \'Cancelled\' and paymentamount > 1)
,providers as (
select * from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_STATUS
where providercreatedon >= \'2012-01-01\'
)
select p1.*, p2.providercreatedon
,CASE WHEN p1.productline = \'Check\' THEN p1.paymentcount ELSE 0 END AS checkpaymentcount
,CASE WHEN p1.productline = \'Check\' THEN p1.paymentamount ELSE 0 END AS checkpaymentamount
,CASE WHEN p1.productline = \'VCC\' THEN p1.paymentcount ELSE 0 END AS vccpaymentcount
,CASE WHEN p1.productline = \'VCC\' THEN p1.paymentamount ELSE 0 END AS vccpaymentamount
,CASE WHEN p1.productline = \'ACH+\' THEN p1.paymentcount ELSE 0 END AS achpaymentcount
,CASE WHEN p1.productline = \'ACH+\' THEN p1.paymentamount ELSE 0 END AS achpaymentamount
,CASE WHEN p1.productline = \'PayerSponsoredACH\' THEN p1.paymentcount ELSE 0 END AS payersponsoredpaymentcount
,CASE WHEN p1.productline = \'PayerSponsoredACH\' THEN p1.paymentamount ELSE 0 END AS payersponsoredpaymentamount
,CASE WHEN p1.productline = \'ACH Reversal\' THEN p1.paymentcount ELSE 0 END AS reversalpaymentcount
,CASE WHEN p1.productline = \'ACH Reversal\' THEN p1.paymentamount ELSE 0 END AS reversalpaymentamount
from payments p1
inner join providers p2
on p1.TIN = p2.TIN and p1.providerid = p2.providerid
);

--##############################################################################################################
-- This table contains the eligible providers
--##############################################################################################################
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF as (
with provider_payments_firstach as (
select TIN, providerid, min(paymentweek) as FIRSTACHPAYDATE from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF
where productline = \'ACH+\'
group by tin, providerid
), 
provider_payments_firstvcc as (
select TIN, providerid, min(paymentweek) as FIRSTVCCPAYDATE from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF
where productline = \'VCC\'
group by tin, providerid
),
optin_ach_info as (
select tin, providerid, min(statusstartdate) as DATE_FIRST_ACH_OPTIN, 
max(statusstartdate) as DATE_LAST_ACH_OPTIN, 
COUNT(DISTINCT CASE WHEN statusname = \'Opt-In\' and productlinename like \'%VRA%\' THEN TO_DATE(statusstartdate)
                   END) AS ACH_OPTIN_COUNT
from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY
where statusname = \'Opt-In\' and productlinename like \'%VRA%\'
group by tin, providerid
),
optin_vcc_info as (
select tin, providerid, min(statusstartdate) as DATE_FIRST_VCC_OPTIN, 
max(statusstartdate) as DATE_LAST_VCC_OPTIN,
COUNT(DISTINCT CASE WHEN statusname = \'Opt-In\' and productlinename like \'%Select%\' THEN TO_DATE(statusstartdate)
                   END) AS VCC_OPTIN_COUNT
from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY
where statusname = \'Opt-In\' and productlinename like \'%Select%\'
group by tin, providerid
),
optout_info as (
select tin, providerid,  
COUNT(DISTINCT CASE WHEN statusname = \'Opt-Out\' THEN TO_DATE(statusstartdate)
                   END) AS OPTOUT_COUNTS,
min(statusstartdate) as DATE_FIRST_OPTOUT, 
max(statusstartdate) as date_last_optout 
from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY
where statusname = \'Opt-Out\'
group by tin, providerid
),
eligible_providers as (
select * from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_STATUS
where providercreatedon >= \'2012-01-01\'
)
select e.*, ach.firstachpaydate as firstachpaydate
, TO_DATE(a.DATE_FIRST_ACH_OPTIN) as DATE_FIRST_ACH_OPTIN
, TO_DATE(a.DATE_LAST_ACH_OPTIN) as DATE_LAST_ACH_OPTIN
, a.ACH_OPTIN_COUNT
, case when firstachpaydate is null then 0 else 1 end as OPTIN_ACH --not final target
, vcc.firstvccpaydate as firstvccpaydate
, TO_DATE(v.DATE_FIRST_VCC_OPTIN) AS DATE_FIRST_VCC_OPTIN
, TO_DATE(v.DATE_LAST_VCC_OPTIN) AS DATE_LAST_VCC_OPTIN
, v.VCC_OPTIN_COUNT
, case when firstvccpaydate is null then 0 else 1 end as OPTIN_VCC --not final target
, o.optout_counts, o.date_first_optout, o.date_last_optout
from eligible_providers e
left join provider_payments_firstach ach
on e.tin = ach.tin and e.providerid = ach.providerid
left join provider_payments_firstvcc vcc
on e.tin = vcc.tin and e.providerid = vcc.providerid
left join optout_info o
on e.tin = o.tin and e.providerid = o.providerid
left join optin_ach_info a
on e.tin = a.tin and e.providerid = a.providerid
left join optin_vcc_info v
on e.tin = v.tin and e.providerid = v.providerid
);


--##############################################################################################################
-- First step table that contains ACH opt-out eligible providers
--##############################################################################################################
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.SILVER_ACH_OPTOUT_0_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF as (

select *, providercreatedon as DATE_START, LEAST(firstachpaydate, date_first_ach_optin) as DATE_END,
DATE_END - DATE_START as time_of_study, OPTIN_ACH as OPTIN_TARGET_ACH,
OPTIN_VCC as OPTIN_TARGET_VCC, \'Provider All Time Opt Outs = 0\' as SILVER_COHORT
from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
where optout_counts is null and optin_ach = 1 and optin_vcc = 0
union

select *, providercreatedon as DATE_START, LEAST(firstvccpaydate, date_first_vcc_optin) as DATE_END,
DATE_END - DATE_START as time_of_study, OPTIN_ACH as OPTIN_TARGET_ACH,
OPTIN_VCC as OPTIN_TARGET_VCC, \'Provider All Time Opt Outs = 0\' as SILVER_COHORT
from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
where optout_counts is null and optin_vcc = 1 and optin_ach = 0
union

select *, providercreatedon as DATE_START
, case when LEAST(firstachpaydate, date_first_ach_optin) < LEAST(firstvccpaydate, date_first_vcc_optin) then LEAST(firstachpaydate, date_first_ach_optin) else LEAST(firstvccpaydate, date_first_vcc_optin) end as DATE_END
, DATE_END - DATE_START as time_of_study
, case when LEAST(firstachpaydate, date_first_ach_optin) <= LEAST(firstvccpaydate, date_first_vcc_optin) then 1 else 0 end as OPTIN_TARGET_ACH
, case when LEAST(firstvccpaydate, date_first_vcc_optin) <= LEAST(firstachpaydate, date_first_ach_optin) then 1 else 0 end as OPTIN_TARGET_VCC
, \'Provider All Time Opt Outs = 0\' as SILVER_COHORT
from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
where optout_counts is null and optin_vcc = 1 and optin_ach = 1
union

select *, providercreatedon as DATE_START, CURRENT_DATE() as DATE_END
, DATE_END - DATE_START as time_of_study
,0 as OPTIN_TARGET_ACH
,0 as OPTIN_TARGET_VCC
, \'Provider All Time Opt Outs = 0\' as SILVER_COHORT
from dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
where optout_counts is null and optin_vcc = 0 and optin_ach = 0
);


--##############################################################################################################
-- Second step table that contains ACH opt-out eligible providers
--##############################################################################################################
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.SILVER_ACH_OPTOUT_1_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF as (
WITH DateofInterest AS (
    -- Get account creation snapshots
    SELECT 
        concat(e.tin, \' | \', e.providerid) as TPID,
        e.TIN,
        e.providerid,
        e.providercreatedon AS ActionDate,
        \'Account Creation\' AS ActionType,
        null as productline
    FROM dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    where e.optout_counts = 1

    UNION ALL

    -- Get Opt-Out snapshots
    SELECT
        concat(h.tin, \' | \', h.providerid) as TPID,
        h.TIN,
        h.providerid,
        h.statusstartdate AS ActionDate,
        \'Opt-Out\' AS ActionType,
        null as productline
    FROM PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY h
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e 
    ON e.TIN = h.TIN and e.providerid = h.providerid -- Ensure only people of interest are included
    WHERE h.statusname = \'Opt-Out\' and e.optout_counts = 1
    
    UNION ALL

    -- Get Opt-In snapshots through Payments view
    SELECT 
        concat(p.tin, \' | \', p.providerid) as TPID,
        p.TIN,
        p.providerid,
        p.paymentweek as ActionDate,
        \'Opt-In\' as ActionType,
        case when p.productline = \'ACH+\' then \'ACH+\' else \'VCC\' end as productline
    FROM dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF p
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    on e.tin = p.tin and e.providerid = p.providerid
    where p.productline = \'ACH+\' or p.productline = \'VCC\' and e.optout_counts = 1

    UNION ALL

    -- Get Opt-In snapshots through Status view
    SELECT 
        concat(h.tin, \' | \', h.providerid) as TPID,
        h.TIN,
        h.providerid,
        TO_DATE(h.statusstartdate) as ActionDate,
        \'Opt-In\' as ActionType,
        case when h.productlinename like \'%VRA%\' then \'ACH+\' else \'VCC\' end as productline
    FROM PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY h
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    on e.tin = h.tin and e.providerid = h.providerid
    where h.statusname = \'Opt-In\' and (h.productlinename like \'%VRA%\' or h.productlinename like \'%Select%\') and e.optout_counts = 1
    
)
-- Put all the snapshots together and chronologically find what comes next
, RankedActions AS (
    -- Rank the actions in order for each person
    SELECT
        TPID, tin, providerid,
        ActionDate,
        ActionType,
        LEAD(ActionDate) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextActionDate,
        LEAD(ActionType) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextActionType,
        LEAD(productline) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextProductLine,
    FROM DateofInterest
)
-- If next date is null, use today\'s date. This will be the right-censored observation.
, tmp_table as (
SELECT 
    TPID, TIN, PROVIDERID,
    ActionDate AS StartDate,
    COALESCE(NextActionDate, CURRENT_DATE()) AS EndDate,  -- Use a future date if there is no next action
    ActionType AS StartStatus,
    NextActionType AS EndStatus,
    NextProductLine
FROM RankedActions
WHERE 
    (ActionType = \'Opt-Out\' AND NextActionType = \'Opt-In\')  -- Periods of Opt-Out to next Opt-In
    OR (ActionType = \'Opt-Out\' AND NextActionType is null) -- This is when Opt-Out happens and no action until present day.
    OR  ActionType = \'Account Creation\'  -- Include the first period from account creation
    -- Not including action type = opt out and next action type = opt out bc this is the case where opt out counts = 1
ORDER BY TPID, StartDate
)
, optout_1_table as (
SELECT 
        t.TPID, t.TIN, t.PROVIDERID, e.SEGMENT, e.APV, e.PROVIDERTYPE, e.TAXONOMYGENERAL, e.TAXONOMYSPECIALTY, e.STATE, e.PROVIDERCREATEDON,
        TO_DATE(t.StartDate) AS DATE_START,
        TO_DATE(t.EndDate) AS DATE_END,
        t.StartStatus,
        t.EndStatus,
        t.NextProductLine,

        SUM(CASE WHEN t.EndStatus = \'Opt-In\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousOptInCount,

        SUM(CASE WHEN EndStatus = \'Opt-In\' AND t.NextProductLine = \'ACH+\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousACHOptInCount,

        SUM(CASE WHEN EndStatus = \'Opt-In\' AND t.NextProductLine = \'VCC\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousVCCOptInCount,
        LAG(t.StartDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousStartDate,
        LAG(t.StartStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousStartStatus,
        LAG(t.EndDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousEndDate,
        LAG(t.EndStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousEndStatus,
        LEAD(t.StartDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS NextStartDate,
        LEAD(t.StartStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS NextStartStatus,
        
        CASE WHEN PreviousEndStatus = \'Opt-In\' THEN DATEDIFF(day, PreviousEndDate, t.StartDate) ELSE NULL END AS LastOptInDuration,
        DATE_END - DATE_START as time_of_study,        
        case when endstatus = \'Opt-In\' and nextproductline = \'ACH+\' then 1 else 0 end as OPTIN_TARGET_ACH,
        case when endstatus = \'Opt-In\' and nextproductline = \'VCC\' then 1 else 0 end as OPTIN_TARGET_VCC
FROM tmp_table t
left join dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
on t.tin = e.tin and t.providerid = e.providerid 
order by t.providerid, date_start
)
, bool_ach_payments_during_optin as(
SELECT t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end,
    coalesce(SUM(t2.achpaymentcount), 0) AS ach_counts_during_optin
FROM optout_1_table t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND t2.paymentweek between t1.date_end AND coalesce(t1.nextstartdate, CURRENT_DATE())
where t2.productline = \'ACH+\'
GROUP BY 
    t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
)
, bool_vcc_payments_during_optin as(
SELECT t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end,
    coalesce(SUM(t2.vccpaymentcount), 0) AS vcc_counts_during_optin
FROM optout_1_table t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND t2.paymentweek between t1.date_end AND coalesce(t1.nextstartdate, CURRENT_DATE())
where t2.productline = \'VCC\'
GROUP BY 
    t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
)
select o1.TPID, o1.TIN, o1.PROVIDERID, o1.SEGMENT, o1.APV, o1.PROVIDERTYPE, o1.TAXONOMYGENERAL, o1.TAXONOMYSPECIALTY, o1.STATE, o1.PROVIDERCREATEDON, o1.DATE_START, o1.DATE_END,
o1.PREVIOUSOPTINCOUNT, o1.PREVIOUSACHOPTINCOUNT, o1.PREVIOUSVCCOPTINCOUNT, o1.LASTOPTINDURATION, o1.TIME_OF_STUDY, o1.OPTIN_TARGET_ACH, o1.OPTIN_TARGET_VCC, o2.ach_counts_during_optin, o3.vcc_counts_during_optin
, \'Provider All Time Opt Outs = 1\' as SILVER_COHORT
, case when (o1.OPTIN_TARGET_ACH = 0 and o1.OPTIN_TARGET_VCC = 0)
            or (o1.OPTIN_TARGET_ACH = 1 and o2.ach_counts_during_optin is not NULL)
            or (o1.OPTIN_TARGET_VCC = 1 and o3.vcc_counts_during_optin is not NULL)
            then 1 else 0 end as semi_row_eligible
, case
    when semi_row_eligible = 1 then o1.optin_target_ach
    when (semi_row_eligible = 0 and o1.OPTIN_TARGET_VCC = 1 and o2.ach_counts_during_optin is not null) then 1
    else 0 end as new_optin_target_ach
--this is okay to do bc this logic says that the provider hasn\'t even tried vcc before jumping to ach+. Most often, the Optin to ach+ happens the 
--same day or the next day. This isn\'t a true upgrade, but rather a quick change of mind.
, case 
    when semi_row_eligible = 1 then o1.optin_target_vcc
    when (semi_row_eligible = 0 and o1.optin_target_ach = 1 and o3.vcc_counts_during_optin is not null) then 1 
    else 0 end as new_optin_target_vcc

from optout_1_table o1
left join bool_ach_payments_during_optin o2
on o1.tin = o2.tin and o1.providerid = o2.providerid and o1.date_start = o2.date_start and o1.date_end = o2.date_end
left join bool_vcc_payments_during_optin o3
on o1.tin = o3.tin and o1.providerid = o3.providerid and o1.date_start = o3.date_start and o1.date_end = o3.date_end
);


--##############################################################################################################
-- Third step table that contains ACH opt-out eligible providers
--##############################################################################################################
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.SILVER_ACH_OPTOUT_GT_1_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF as (
WITH DateofInterest AS (
    -- Get OptIn actions
    SELECT 
        concat(e.tin, \' | \', e.providerid) as TPID,
        e.TIN,
        e.providerid,
        e.providercreatedon AS ActionDate,
        \'Account Creation\' AS ActionType,
        null as productline
    FROM dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    where e.optout_counts > 1

    UNION ALL
    
    SELECT
        concat(h.tin, \' | \', h.providerid) as TPID,
        h.TIN,
        h.providerid,
        h.statusstartdate AS ActionDate,
        \'Opt-Out\' AS ActionType,
        null as productline
    FROM PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY h
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e 
    ON e.TIN = h.TIN and e.providerid = h.providerid -- Ensure only people of interest are included
    WHERE h.statusname = \'Opt-Out\' and e.optout_counts > 1
    
    UNION ALL
    
    SELECT 
        concat(p.tin, \' | \', p.providerid) as TPID,
        p.TIN,
        p.providerid,
        p.paymentweek as ActionDate,
        \'Opt-In\' as ActionType,
        case when p.productline = \'ACH+\' then \'ACH+\' else \'VCC\' end as productline
    FROM dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF p
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    on e.tin = p.tin and e.providerid = p.providerid
    where p.productline = \'ACH+\' or p.productline = \'VCC\' and e.optout_counts > 1

    UNION ALL

    SELECT 
        concat(h.tin, \' | \', h.providerid) as TPID,
        h.TIN,
        h.providerid,
        TO_DATE(h.statusstartdate) as ActionDate,
        \'Opt-In\' as ActionType,
        case when h.productlinename like \'%VRA%\' then \'ACH+\' else \'VCC\' end as productline
    FROM PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY h
    INNER JOIN dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
    on e.tin = h.tin and e.providerid = h.providerid
    where h.statusname = \'Opt-In\' and (h.productlinename like \'%VRA%\' or h.productlinename like \'%Select%\') and e.optout_counts > 1
)
, RankedActions AS (
    -- Rank the actions in order for each person
    SELECT
        TPID, tin, providerid,
        ActionDate,
        ActionType,
        LEAD(ActionDate) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextActionDate,
        LEAD(ActionType) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextActionType,
        LEAD(productline) OVER (PARTITION BY TPID ORDER BY ActionDate) AS NextProductLine,
    FROM DateofInterest
)
, tmp_table as (
SELECT 
    TPID, TIN, PROVIDERID,
    ActionDate AS StartDate,
    COALESCE(NextActionDate, CURRENT_DATE()) AS EndDate,  -- Use a future date if there is no next action
    ActionType AS StartStatus,
    NextActionType AS EndStatus,
    NextProductLine
FROM RankedActions
WHERE 
    (ActionType = \'Opt-Out\' AND NextActionType = \'Opt-In\')  -- Periods of Opt-Out to next Opt-In
    OR (ActionType = \'Opt-Out\' AND NextActionType is null) 
    OR (ActionType = \'Opt-Out\' AND NextActionType = \'Opt-Out\') -- handles cases where provider Opted-Out and then 6 months later back in campaign and then Opt-Out again
    OR ActionType = \'Account Creation\'  -- Include the first period from account creation
ORDER BY TPID, StartDate
)
,optout_gt_1_table as (
SELECT 
        t.TPID, t.TIN, t.PROVIDERID, e.SEGMENT, e.APV, e.PROVIDERTYPE, e.TAXONOMYGENERAL, e.TAXONOMYSPECIALTY, e.STATE, e.PROVIDERCREATEDON,
        TO_DATE(t.StartDate) AS DATE_START,
        TO_DATE(t.EndDate) AS DATE_END,
        t.StartStatus,
        t.EndStatus,
        t.NextProductLine,

        SUM(CASE WHEN t.EndStatus = \'Opt-In\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousOptInCount,

        SUM(CASE WHEN EndStatus = \'Opt-In\' AND t.NextProductLine = \'ACH+\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousACHOptInCount,

        SUM(CASE WHEN EndStatus = \'Opt-In\' AND t.NextProductLine = \'VCC\' THEN 1 ELSE 0 END) 
            OVER (PARTITION BY t.TPID ORDER BY t.StartDate ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS PreviousVCCOptInCount,
        LAG(t.StartDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousStartDate,
        LAG(t.StartStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousStartStatus,
        LAG(t.EndDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousEndDate,
        LAG(t.EndStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS PreviousEndStatus,
        LEAD(t.StartDate) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS NextStartDate,
        LEAD(t.StartStatus) OVER (PARTITION BY t.TPID ORDER BY t.StartDate) AS NextStartStatus,
        
        CASE WHEN PreviousEndStatus = \'Opt-In\' THEN DATEDIFF(day, PreviousEndDate, t.StartDate) ELSE NULL END AS LastOptInDuration,
        DATE_END - DATE_START as time_of_study,
        case when endstatus = \'Opt-In\' and nextproductline = \'ACH+\' then 1 else 0 end as OPTIN_TARGET_ACH,
        case when endstatus = \'Opt-In\' and nextproductline = \'VCC\' then 1 else 0 end as OPTIN_TARGET_VCC
FROM tmp_table t
left join dev_zada_datascience.pymts_sales_propensity_all_time.BRONZE_M7_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
on t.tin = e.tin and t.providerid = e.providerid
order by t.providerid, date_start
)
, bool_ach_payments_during_optin as(
SELECT t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end,
    coalesce(SUM(t2.achpaymentcount), 0) AS ach_counts_during_optin
FROM optout_gt_1_table t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND t2.paymentweek between t1.date_end AND coalesce(t1.nextstartdate, to_date(CURRENT_DATE()))
where t2.productline = \'ACH+\'
GROUP BY 
    t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
)
, bool_vcc_payments_during_optin as(
SELECT t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end,
    coalesce(SUM(t2.vccpaymentcount), 0) AS vcc_counts_during_optin
FROM optout_gt_1_table t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND t2.paymentweek between t1.date_end AND coalesce(t1.nextstartdate, CURRENT_DATE())
where t2.productline = \'VCC\'
GROUP BY 
    t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.TPID, t1.tin, t1.providerid, t1.date_start, t1.date_end
)
select o1.TPID, o1.TIN, o1.PROVIDERID, o1.SEGMENT, o1.APV, o1.PROVIDERTYPE, o1.TAXONOMYGENERAL, o1.TAXONOMYSPECIALTY, o1.STATE, o1.PROVIDERCREATEDON, o1.DATE_START, o1.DATE_END,
o1.PREVIOUSOPTINCOUNT, o1.PREVIOUSACHOPTINCOUNT, o1.PREVIOUSVCCOPTINCOUNT, o1.LASTOPTINDURATION, o1.TIME_OF_STUDY, o1.OPTIN_TARGET_ACH, o1.OPTIN_TARGET_VCC, o2.ach_counts_during_optin, o3.vcc_counts_during_optin
, \'Provider All Time Opt Outs > 1\' as SILVER_COHORT
, case when (o1.OPTIN_TARGET_ACH = 0 and o1.OPTIN_TARGET_VCC = 0)
            or (o1.OPTIN_TARGET_ACH = 1 and o2.ach_counts_during_optin is not NULL)
            or (o1.OPTIN_TARGET_VCC = 1 and o3.vcc_counts_during_optin is not NULL)
            then 1 else 0 end as semi_row_eligible
, case
    when semi_row_eligible = 1 then o1.optin_target_ach
    when (semi_row_eligible = 0 and o1.OPTIN_TARGET_VCC = 1 and o2.ach_counts_during_optin is not null) then 1
    else 0 end as new_optin_target_ach
--this is okay to do bc this logic says that the provider hasn\'t even tried vcc before jumping to ach+. Most often, the Optin to ach+ happens the 
--same day or the next day. This isn\'t a true upgrade, but rather a quick change of mind.
, case 
    when semi_row_eligible = 1 then o1.optin_target_vcc
    when (semi_row_eligible = 0 and o1.optin_target_ach = 1 and o3.vcc_counts_during_optin is not null) then 1 
    else 0 end as new_optin_target_vcc

from optout_gt_1_table o1
left join bool_ach_payments_during_optin o2
on o1.tin = o2.tin and o1.providerid = o2.providerid and o1.date_start = o2.date_start and o1.date_end = o2.date_end
left join bool_vcc_payments_during_optin o3
on o1.tin = o3.tin and o1.providerid = o3.providerid and o1.date_start = o3.date_start and o1.date_end = o3.date_end
);


CREATE OR REPLACE TABLE DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.tmp_20260223 AS (
-- Generates dates to join to avoid unions
WITH snapshots as (
  SELECT dateadd(day, seq4(), to_date(\'2022-01-01\')) as snap_dt
  FROM TABLE(generator(rowcount => 2000))
  WHERE dateadd(day, seq4(), to_date(\'2022-01-01\')) <= to_date(\'2026-02-01\')
)
, dates as (
SELECT snap_dt
FROM snapshots
WHERE day(snap_dt) in (1, 15) -- 1st and 15th of every month. Change to 1 if only every 30 days.
)
, gold_eligible_providers as (
select concat(TIN, \' | \', PROVIDERID) as TPID, TIN, PROVIDERID, SEGMENT, APV, PROVIDERTYPE, TAXONOMYGENERAL, TAXONOMYSPECIALTY, STATE, 
PROVIDERCREATEDON, DATE_START, DATE_END, 0 as PREVIOUSOPTINCOUNT, 0 as PREVIOUSACHOPTINCOUNT, 0 as PREVIOUSVCCOPTINCOUNT, null as LASTOPTINDURATION, TIME_OF_STUDY, OPTIN_TARGET_ACH, OPTIN_TARGET_VCC
, SILVER_COHORT
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.SILVER_ACH_OPTOUT_0_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
union
select TPID, TIN, PROVIDERID, SEGMENT, APV, PROVIDERTYPE, TAXONOMYGENERAL, TAXONOMYSPECIALTY, STATE,
PROVIDERCREATEDON, DATE_START, DATE_END, previousoptincount, previousachoptincount, previousvccoptincount,
lastoptinduration, time_of_study, new_optin_target_ach as optin_target_ach, new_optin_target_vcc as optin_target_vcc
, SILVER_COHORT
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.SILVER_ACH_OPTOUT_1_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
union
select TPID, TIN, PROVIDERID, SEGMENT, APV, PROVIDERTYPE, TAXONOMYGENERAL, TAXONOMYSPECIALTY, STATE,
PROVIDERCREATEDON, DATE_START, DATE_END, previousoptincount, previousachoptincount, previousvccoptincount,
lastoptinduration, time_of_study, new_optin_target_ach as optin_target_ach, new_optin_target_vcc as optin_target_vcc
, SILVER_COHORT
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.SILVER_ACH_OPTOUT_GT_1_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
)

SELECT distinct s.*,
  d.snap_dt as DATE_END_OOS,
  d.snap_dt - s.DATE_START as DURATION_OOS
FROM gold_eligible_providers s
JOIN dates d ON d.snap_dt > s.DATE_START AND d.snap_dt <= s.DATE_END
order by s.tpid, duration_oos
);

CREATE OR REPLACE TABLE DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.silver_ach_optin_from_paper_tmp AS 
(
SELECT t.TPID, t.TIN, t.PROVIDERID, t.SEGMENT, t.APV, t.PROVIDERTYPE, t.TAXONOMYGENERAL, t.TAXONOMYSPECIALTY, t.STATE, t.PROVIDERCREATEDON
    , t.DATE_START, t.DATE_END_OOS as DATE_END, t.DURATION_OOS as DURATION, DATEADD(DAY, 30, t.date_end_oos) as DATE_END_PLUS_30
    , CASE WHEN (t.DATE_END BETWEEN t.date_end_oos and DATEADD(DAY, 30, t.date_end_oos)) and (t.optin_target_ach = 1) THEN 1 ELSE 0 END as opt_in_ach
    , CASE WHEN (t.DATE_END BETWEEN t.date_end_oos and DATEADD(DAY, 30, t.date_end_oos)) and (t.optin_target_vcc = 1) THEN 1 ELSE 0 END as opt_in_vcc
FROM DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.tmp_20260223 t
);

--##############################################################################################################
-- GOLD ACH eligible providers table
--##############################################################################################################
-- Want to only focus on providers with duration > 90 because:
-- 1. for undecideds, <90 is a different model
-- 2. for circle backs, need to wait 6M-9M to reach out again after the Opt-Out;
-- Therefore, NO providers will be reached out until at least 90 days (undecideds) have passed
-- 3. 90 days provides enough time for feature building
create or replace table dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF as (
select *
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.silver_ach_optin_from_paper_tmp
where duration > 90
);


--##############################################################################################################
-- Final GOLD PSR4 ACH Train table
--##############################################################################################################

create or replace table IDENTIFIER(:final_table_name) AS (
with check_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.checkpaymentcount) AS sumcheckpaymentcount,
    SUM(t2.checkpaymentamount) AS sumcheckpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_check
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
),
check_cancelled_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.checkpaymentcount) AS cancelledcheckpaymentcount,
    SUM(t2.checkpaymentamount) AS cancelledcheckpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as cancelled_payeridlist_check
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
),
last_cancelled_check_payment as (
select t2.tin, t2.providerid, t2.date_start, t2.date_end
, MAX (T1.PAYMENTWEEK) AS LASTCANCELLEDCHECKDATE
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF T1
left join dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF T2 ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
where to_date(t1.paymentweek) >= to_date(t2.date_start) AND to_date(t1.paymentweek) < to_date(t2.date_end)
and T1.STATUSNAME = \'Cancelled\' and T1.productline = \'Check\'
group by t2.tin, t2.providerid, t2.date_start, t2.date_end
),
last_check_payment as (
select t2.tin, t2.providerid, t2.date_start, t2.date_end
, MAX (T1.PAYMENTWEEK) AS LASTCHECKDATE
from DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF T1
left join dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF T2 ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
where to_date(t1.paymentweek) >= to_date(t2.date_start) AND to_date(t1.paymentweek) < to_date(t2.date_end)
and T1.STATUSNAME = \'Processed - Paid\' and T1.productline = \'Check\'
group by t2.tin, t2.providerid, t2.date_start, t2.date_end
),
check_payments30 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS sumcheckpaymentcount30,
    SUM(t2.checkpaymentamount) AS sumcheckpaymentamount30
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -30,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
check_payments60 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS sumcheckpaymentcount60,
    SUM(t2.checkpaymentamount) AS sumcheckpaymentamount60
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -60,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
check_payments90 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS sumcheckpaymentcount90,
    SUM(t2.checkpaymentamount) AS sumcheckpaymentamount90
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -90,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
check_cancelled_payments30 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS cancelledcheckpaymentcount30,
    SUM(t2.checkpaymentamount) AS cancelledcheckpaymentamount30
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -30,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
check_cancelled_payments60 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS cancelledcheckpaymentcount60,
    SUM(t2.checkpaymentamount) AS cancelledcheckpaymentamount60
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -60,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
check_cancelled_payments90 as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_end,
    SUM(t2.checkpaymentcount) AS cancelledcheckpaymentcount90,
    SUM(t2.checkpaymentamount) AS cancelledcheckpaymentamount90
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_CANCELLED_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= DATEADD(day, -90,t1.date_end) AND to_date(t2.paymentweek) < to_date(t1.date_end)
where t2.productline = \'Check\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_end
),
vcc_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.vccpaymentcount) AS sumvccpaymentcount,
    SUM(t2.vccpaymentamount) AS sumvccpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_vcc
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'VCC\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
), ach_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.achpaymentcount) AS sumachpaymentcount,
    SUM(t2.achpaymentamount) AS sumachpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_ach
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'ACH+\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
), sponsored_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.payersponsoredpaymentcount) AS sumpayersponsoredpaymentcount,
    SUM(t2.payersponsoredpaymentamount) AS sumpayersponsoredpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_payersponsored
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'PayerSponsoredACH\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
), reversal_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.reversalpaymentcount) AS sumreversalpaymentcount,
    SUM(t2.reversalpaymentamount) AS sumreversalpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_reversal
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'ACH Reversal\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
), total_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.paymentcount) AS sumtotalpaymentcount,
    SUM(t2.paymentamount) AS sumtotalpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_total
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline != \'ACH Reversal\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
), important_payments as (
SELECT t1.TIN, t1.PROVIDERID, t1.date_start, t1.date_end,
    SUM(t2.paymentcount) AS sumimportantpaymentcount,
    SUM(t2.paymentamount) AS sumimportantpaymentamount,
    listagg(distinct t2.payerid, \', \') within group (order by t2.payerid) as payerid_list_important
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
LEFT JOIN DEV_ZADA_DATASCIENCE.PYMTS_SALES_PROPENSITY_ALL_TIME.BRONZE_PROVIDER_PYMTS_2012_OR_NEWER_RF t2
    ON t1.TIN = t2.TIN and t1.providerid = t2.providerid
    AND to_date(t2.paymentweek) >= to_date(t1.date_start) AND to_date(t2.paymentweek) < to_date(t1.date_end) -- end date exclusive bc it\'s additional layer to prevent data leakage
where t2.productline = \'Check\' or t2.productline = \'PayerSponsoredACH\'
GROUP BY 
    t1.tin, t1.providerid, t1.date_start, t1.date_end
ORDER BY t1.tin, t1.providerid, t1.date_start, t1.date_end
)
, optouts as (
SELECT 
    t2.tin,
    t2.providerid,
    t2.date_end, 
    COALESCE(COUNT(t1.tin), 0) AS total_prev_optouts,
    MAX(t1.statusstartdate) AS date_last_optout,
    MAX_BY(t1.notes, t1.statusstartdate) AS last_optout_note,
    IFF(
      COALESCE(
        BOOLOR_AGG(
          LOWER(COALESCE(t1.notes, \'\')) LIKE \'%fee%\'
          OR LOWER(COALESCE(t1.notes, \'\')) LIKE \'%want to pay to be paid%\'
        ),
        FALSE
      ), 1, 0) AS optout_notes_fee_flag,
    IFF(
      COALESCE(
        BOOLOR_AGG(
          LOWER(COALESCE(t1.notes, \'\')) LIKE \'%understand%\'
          OR LOWER(COALESCE(t1.notes, \'\')) LIKE \'%agree to product%\'
        ),
        FALSE
      ), 1, 0) AS optout_notes_dont_understand_agree_flag   
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t2
LEFT JOIN PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY t1 
    ON t1.tin = t2.tin
    and t1.providerid = t2.providerid
    AND t1.statusname = \'Opt-Out\'
    and t1.optout_reason = 0
    and t1.statusstartdate < t2.date_end
GROUP BY t2.tin, t2.providerid, t2.date_end
)
, optouts_last_year as (
SELECT 
    t2.tin,
    t2.providerid,
    t2.date_end, 
    COALESCE(COUNT(t1.tin), 0) AS total_prev_optouts_last_year,
    MAX(t1.statusstartdate) AS date_last_optout_last_year,
    MAX_BY(t1.notes, t1.statusstartdate) AS last_optout_note_last_year,
    IFF(
      COALESCE(
        BOOLOR_AGG(
          LOWER(COALESCE(t1.notes, \'\')) LIKE \'%fee%\'
          OR LOWER(COALESCE(t1.notes, \'\')) LIKE \'%want to pay to be paid%\'
        ),
        FALSE
      ), 1, 0) AS optout_notes_fee_flag_last_year,
    IFF(
      COALESCE(
        BOOLOR_AGG(
          LOWER(COALESCE(t1.notes, \'\')) LIKE \'%understand%\'
          OR LOWER(COALESCE(t1.notes, \'\')) LIKE \'%agree to product%\'
        ),
        FALSE
      ), 1, 0) AS optout_notes_dont_understand_agree_flag_last_year   
FROM dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t2
LEFT JOIN PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_HISTORY t1 
    ON t1.tin = t2.tin
    and t1.providerid = t2.providerid
    AND t1.statusname = \'Opt-Out\'
    and t1.optout_reason = 0
    AND t1.statusstartdate >= DATEADD(day, -365,t2.date_end)
    and t1.statusstartdate < t2.date_end
GROUP BY t2.tin, t2.providerid, t2.date_end
)
, unique_providers as (
select distinct tin, providerid
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF
)
, provider_interactions as (
select i.*
from PROD_ZDI_DSE.ZDI_BATCH_MODELS.PROVIDER_INTERACTIONS i
inner join unique_providers e
on e.TIN = i.TIN and e.PROVIDERID = i.PROVIDERID
)
, phone_count as (
SELECT t1.TIN, t1.PROVIDERID, t1.DATE_START, t1.DATE_END,
SUM(
    CASE
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'1099 Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Assisted with Registration\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Cancel Check\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'EOP Discrepancy\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'ePayment Center\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escalation\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escheatment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Exclusion Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Member\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'New Check Issued\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Not receiving Remits\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payer List Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payment Already Processed\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Provisioning Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred Provider to Payer\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to ePC Portal\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to Zelis Health Care\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Resent EOP\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Stop Payment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Zelis Client Service\') THEN 1

        ELSE 0
    END
) AS PHONECOUNT
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
left join provider_interactions i
    ON t1.TIN = i.TIN and t1.providerid = i.providerid
    AND to_date(i.interactiondate) >= to_date(t1.date_start) AND to_date(i.interactiondate) < to_date(t1.date_end)
GROUP BY t1.TIN, t1.PROVIDERID, t1.DATE_START, t1.DATE_END
), phone_count30 as (
SELECT t1.TIN, t1.PROVIDERID, t1.DATE_END,
SUM(
    CASE
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'1099 Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Assisted with Registration\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Cancel Check\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'EOP Discrepancy\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'ePayment Center\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escalation\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escheatment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Exclusion Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Member\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'New Check Issued\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Not receiving Remits\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payer List Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payment Already Processed\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Provisioning Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred Provider to Payer\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to ePC Portal\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to Zelis Health Care\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Resent EOP\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Stop Payment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Zelis Client Service\') THEN 1

        ELSE 0
    END
) AS PHONECOUNT30
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
left join provider_interactions i
    ON t1.TIN = i.TIN and t1.providerid = i.providerid
    AND to_date(i.interactiondate) >= DATEADD(day, -30,t1.date_end) AND to_date(i.interactiondate) < to_date(t1.date_end)
GROUP BY t1.TIN, t1.PROVIDERID, t1.DATE_END
),
phone_count60 as (
SELECT t1.TIN, t1.PROVIDERID, t1.DATE_END,
SUM(
    CASE
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'1099 Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Assisted with Registration\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Cancel Check\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'EOP Discrepancy\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'ePayment Center\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escalation\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escheatment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Exclusion Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Member\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'New Check Issued\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Not receiving Remits\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payer List Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payment Already Processed\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Provisioning Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred Provider to Payer\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to ePC Portal\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to Zelis Health Care\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Resent EOP\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Stop Payment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Zelis Client Service\') THEN 1

        ELSE 0
    END
) AS PHONECOUNT60
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
left join provider_interactions i
    ON t1.TIN = i.TIN and t1.providerid = i.providerid
    AND to_date(i.interactiondate) >= DATEADD(day, -60,t1.date_end) AND to_date(i.interactiondate) < to_date(t1.date_end)
GROUP BY t1.TIN, t1.PROVIDERID, t1.DATE_END
),
phone_count90 as (
SELECT t1.TIN, t1.PROVIDERID, t1.DATE_END,
SUM(
    CASE
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'1099 Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Assisted with Registration\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Cancel Check\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'EOP Discrepancy\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'ePayment Center\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escalation\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Escheatment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Exclusion Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Member\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'New Check Issued\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Not receiving Remits\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payer List Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Payment Already Processed\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Provisioning Request\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred Provider to Payer\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to ePC Portal\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Referred to Zelis Health Care\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Resent EOP\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Stop Payment\') THEN 1
        WHEN (actionname = \'Disposition\' AND actionreasonname = \'Zelis Client Service\') THEN 1

        ELSE 0
    END
) AS PHONECOUNT90
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF t1
left join provider_interactions i
    ON t1.TIN = i.TIN and t1.providerid = i.providerid
    AND to_date(i.interactiondate) >= DATEADD(day, -90,t1.date_end) AND to_date(i.interactiondate) < to_date(t1.date_end)
GROUP BY t1.TIN, t1.PROVIDERID, t1.DATE_END
),
exclude_provider_list as (
Select ml.TIN, ml.ProviderID,DONOTCALL,GLOBALEXCLUDED,
case when StatusID = 31 and productlineid = 3 and (iswhitelabel = \'true\' or isdirectzero = \'true\') then \'true\' else false end as PayerSponsored
from PROD_LEGACYPPSWAREHOUSE.DBO.VWMAP_LAP ml
join PROD_LEGACYPPSWAREHOUSE.DBO.PROVIDER P on p.providerid = ml.providerid
)
select e.*
, c.sumcheckpaymentcount, c.sumcheckpaymentamount, c.payerid_list_check, l.LASTCHECKDATE
, e.date_end - l.lastcheckdate as TIME_SINCE_LAST_CHECK_PAYMENT
, lccp.lastcancelledcheckdate, e.date_end - lccp.lastcancelledcheckdate as TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT
, ccp.cancelledcheckpaymentcount, ccp.cancelledcheckpaymentamount
, ccp30.cancelledcheckpaymentcount30, ccp30.cancelledcheckpaymentamount30
, ccp60.cancelledcheckpaymentcount60, ccp60.cancelledcheckpaymentamount60
, ccp90.cancelledcheckpaymentcount90, ccp90.cancelledcheckpaymentamount90
, c30.sumcheckpaymentcount30, c30.sumcheckpaymentamount30
, c60.sumcheckpaymentcount60, c60.sumcheckpaymentamount60
, c90.sumcheckpaymentcount90, c90.sumcheckpaymentamount90
, a.sumachpaymentcount, a.sumachpaymentamount, a.payerid_list_ach
, v.sumvccpaymentcount, v.sumvccpaymentamount, v.payerid_list_vcc
, s.sumpayersponsoredpaymentcount, s.sumpayersponsoredpaymentamount, s.payerid_list_payersponsored
, r.sumreversalpaymentcount, r.sumreversalpaymentamount, r.payerid_list_reversal
, t.sumtotalpaymentcount, t.sumtotalpaymentamount, t.payerid_list_total
, imp.sumimportantpaymentcount, imp.sumimportantpaymentamount, imp.payerid_list_important
, o.total_prev_optouts, o.date_last_optout, o.last_optout_note, o.optout_notes_fee_flag, o.optout_notes_dont_understand_agree_flag
, oly.total_prev_optouts_last_year, oly.date_last_optout_last_year, oly.last_optout_note_last_year, oly.optout_notes_fee_flag_last_year, oly.optout_notes_dont_understand_agree_flag_last_year
, i.phonecount, i30.phonecount30, i60.phonecount60, i90.phonecount90
, case when imp.payerid_list_important like \'%7376%\' then 1 else 0 end as IMP_CONTAINS_METLIFE
, case when imp.payerid_list_important = \'7376\' then 1 else 0 end as IMP_IS_ONLY_METLIFE
, ex.DONOTCALL, ex.GLOBALEXCLUDED, ex.PayerSponsored
from dev_zada_datascience.pymts_sales_propensity_all_time.GOLD_ACH_ELIGIBLE_PROVIDERS_2012_OR_NEWER_RF e
left join check_payments c
on e.tin = c.tin and e.providerid = c.providerid and e.date_start = c.date_start and e.date_end = c.date_end
left join check_cancelled_payments ccp
on e.tin = ccp.tin and e.providerid = ccp.providerid and e.date_start = ccp.date_start and e.date_end = ccp.date_end
left join last_check_payment l
on e.tin = l.tin and e.providerid = l.providerid and e.date_start = l.date_start and e.date_end = l.date_end
left join check_payments30 c30
on e.tin = c30.tin and e.providerid = c30.providerid and e.date_end = c30.date_end
left join check_payments60 c60
on e.tin = c60.tin and e.providerid = c60.providerid and e.date_end = c60.date_end
left join check_payments90 c90
on e.tin = c90.tin and e.providerid = c90.providerid and e.date_end = c90.date_end
left join check_cancelled_payments30 ccp30
on e.tin = ccp30.tin and e.providerid = ccp30.providerid and e.date_end = ccp30.date_end
left join check_cancelled_payments60 ccp60
on e.tin = ccp60.tin and e.providerid = ccp60.providerid and e.date_end = ccp60.date_end
left join check_cancelled_payments90 ccp90
on e.tin = ccp90.tin and e.providerid = ccp90.providerid and e.date_end = ccp90.date_end
left join last_cancelled_check_payment lccp
on e.tin = lccp.tin and e.providerid = lccp.providerid and e.date_start = lccp.date_start and e.date_end = lccp.date_end
left join vcc_payments v
on e.tin = v.tin and e.providerid = v.providerid and e.date_start = v.date_start and e.date_end = v.date_end
left join ach_payments a
on e.tin = a.tin and e.providerid = a.providerid and e.date_start = a.date_start and e.date_end = a.date_end
left join sponsored_payments s
on e.tin = s.tin and e.providerid = s.providerid and e.date_start = s.date_start and e.date_end = s.date_end
left join reversal_payments r
on e.tin = r.tin and e.providerid = r.providerid and e.date_start = r.date_start and e.date_end = r.date_end
left join total_payments t
on e.tin = t.tin and e.providerid = t.providerid and e.date_start = t.date_start and e.date_end = t.date_end
left join important_payments imp
on e.tin = imp.tin and e.providerid = imp.providerid and e.date_start = imp.date_start and e.date_end = imp.date_end
left join optouts o
on e.tin = o.tin and e.providerid = o.providerid and e.date_end = o.date_end
left join optouts_last_year oly
on e.tin = oly.tin and e.providerid = oly.providerid and e.date_end = oly.date_end
left join phone_count i
on e.tin = i.tin and e.providerid = i.providerid and e.date_start = i.date_start and e.date_end = i.date_end
left join phone_count30 i30
on e.tin = i30.tin and e.providerid = i30.providerid and e.date_end = i30.date_end
left join phone_count60 i60
on e.tin = i60.tin and e.providerid = i60.providerid and e.date_end = i60.date_end
left join phone_count90 i90
on e.tin = i90.tin and e.providerid = i90.providerid and e.date_end = i90.date_end
left join exclude_provider_list ex
on e.tin = ex.tin and e.providerid = ex.providerid
where sumimportantpaymentcount > 0
);

--##############################################################################################################
-- Finalize Stored Procedure
--##############################################################################################################


    -- Return a message
    RETURN result_message;

EXCEPTION\n    WHEN OTHER THEN\n
RETURN \'Failed to create table: \' || SQLSTATE || \' - \' || SQLERRM;\n
END;\n
';