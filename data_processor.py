import threading
from datetime import date
from dateutil.relativedelta import relativedelta

import pandas as pd
import src.athena_query as AQ
from prophet import Prophet
from tqdm.contrib import itertools

VOL_VAR = 'units'
SITE_ID = 'SITE_ID'


class DataProcessor(object):

    def __init__(self, s3_client, s3_resource, athena_client, tgt_var, output_bucket, tgt_val, output_key,
                 query_file, site_var="fc", dt_var='date'):
        """
        Constructor for the data processor.
        Args:
            s3_client: the s3 client to use.
            s3_resource: the s3 resource.
            athena_client: the athena client.
            tgt_var: the target variable.
            output_bucket: the output bucket.
            tgt_val: the target value.
            output_key: the output s3 key.
            conditions: the conditions.
        """
        self.s3_client = s3_client
        self.athena_client = athena_client
        self.site_var = site_var
        self.dt_var = dt_var
        self.s3_resource = s3_resource
        self.tgt_var = tgt_var
        self.start_date = date.today() - relativedelta(months=6)
        self.target = tgt_val
        self.output_key = output_key
        self.output_bucket = output_bucket
        self.query_file = query_file

    def data_prep(self, df, site):
        """
        Data prep function: define extreme values and align format for modeling
        Args:
            df: the dataframe
            site: the site.
        Returns: prepared data.
        """
        # select site
        df = df[df[self.site_var] == site]
        # format date variable
        df[self.dt_var] = df[self.dt_var].apply(pd.to_datetime)
        # make sure the target variable and vol_var is numeric
        df[self.tgt_var] = pd.to_numeric(df[self.tgt_var])
        df[self.tgt_var] = df[self.tgt_var] * 1.00
        # df[vol_var] = df[vol_var]*1.00
        # find extreme values
        upper_bound = df[self.tgt_var].mean() + 4 * df[self.tgt_var].std()
        lower_bound = df[self.tgt_var].mean() - 4 * df[self.tgt_var].std()
        extreme_df = df[(df[self.tgt_var] > upper_bound) | (df[self.tgt_var] < lower_bound)]
        df = df[(df[self.tgt_var] <= upper_bound) & (df[self.tgt_var] >= lower_bound)]
        # align the format to prophet requirement
        prpht_prep_df = df[[self.dt_var, self.tgt_var, VOL_VAR]]
        prpht_prep_df.reset_index()
        prpht_prep_df.columns = ['ds', 'y', 'ttl_vol']
        print(prpht_prep_df.head())
        return prpht_prep_df, extreme_df

    def prpht_para_slct(self, model_df):
        """
        Model: prophet model selection
        Args:
            model_df: the model dataframe
        Returns: the best param.

        """
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0]
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmse = []

        for params in all_params:
            m = Prophet(**params, weekly_seasonality=True, interval_width=0.9, changepoint_range=0.95,
                        seasonality_mode='multiplicative')
            m.add_regressor('ttl_vol')
            m.fit(model_df)
            f = m.predict(model_df)
            df_per = pd.merge(model_df, f[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
            rmse.append(((df_per.y - df_per.yhat) ** 2).mean() ** .5)

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmse
        best_param = tuning_results[tuning_results.rmse == tuning_results.rmse.min()]
        print(best_param)
        return best_param

    def mdl_fit(self, model_df, best_param):
        """
        Model: build using the best parameter set selected
        Args:
            model_df: the model dataframe.
            best_param: the best param.
        Returns: model and perf dataframe.
        """
        model = Prophet(changepoint_prior_scale=best_param['changepoint_prior_scale'].values[0],
                        seasonality_prior_scale=best_param['seasonality_prior_scale'].values[0],
                        changepoint_range=0.95,
                        interval_width=0.9,
                        weekly_seasonality=True,
                        seasonality_mode='multiplicative').add_regressor('ttl_vol').fit(model_df)
        forecast_df = model.predict(model_df)
        #Extract Changepoinst
        changepoints=model.changepoints
        deltas=model.params['delta'][0]
        perf_df = pd.merge(model_df, forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
        perf_df['anomaly_hi'] = perf_df.apply(lambda rows: 1 if (rows.y > rows.yhat_upper) else 0, axis=1)
        perf_df['anomaly_lw'] = perf_df.apply(lambda rows: 1 if (rows.y < rows.yhat_lower) else 0, axis=1)
        perf_df['anomaly'] = perf_df.apply(
            lambda rows: 1 if ((rows.y > rows.yhat_upper) | (rows.y < rows.yhat_lower)) else 0, axis=1)
        for change_date, delta in zip(changepoints, deltas):
            perf_df.loc[perf_df['ds']==change_date, 'changepoint']=1
            perf_df.loc[perf_df['ds']==change_date, 'delta']=delta
        perf_df.fillna({'changepoint':0, 'delta':0}, inplace=True)
        print(perf_df.head())
        return model, perf_df

    def format_output(self, results_df, site):
        """
        Output: create anomaly data frame
        Args:
            results_df: the resulting dataframe.
            site: the site.
        Returns: the formatted dataframe.
        """
        #output_df = results_df[(results_df['anomaly'] == 1)]
        output_df=results_df
        output_df[self.site_var] = site
        output_df['ds'] = output_df['ds'].apply(pd.to_datetime).dt.date
        output_df['target'] = self.target
        print(output_df.head())
        return output_df
    
    def format_out2(self,model,df,site):
        future_df_dt = model.make_future_dataframe(periods = 30)
        site_df = df[df[self.site_var] == site]
        future_df_vol = site_df[[self.dt_var,VOL_VAR]]
        future_df_vol[self.dt_var] = future_df_vol[self.dt_var].apply(pd.to_datetime)
        future = future_df_dt.merge(future_df_vol, left_on = 'ds',right_on = self.dt_var, how='left')
        future = future[['ds',VOL_VAR]]
        future.columns = ['ds', 'ttl_vol']
        future['ttl_vol'] = future.apply(lambda x: 0 if pd.isnull(x['ttl_vol']) else x['ttl_vol'], axis=1)
        #prediction
        forecast_future = model.predict(future)
        forecast_output1 = forecast_future.tail(30)
        forecast_output2 = forecast_output1[['ds','yhat','yhat_lower','yhat_upper']]
        #format
        forecast_output2[self.site_var]=site
        forecast_output3 = forecast_output2.merge(future_df_vol, left_on = 'ds',right_on = self.dt_var, how='left')
        forecast_output3 = forecast_output3[[self.site_var,'ds','yhat','yhat_lower','yhat_upper',VOL_VAR]]
        forecast_output3['yhat'] = forecast_output3.apply(lambda x: 0 if pd.isnull(x[VOL_VAR]) else x['yhat'], axis=1)
        print(forecast_output3.head())
        return forecast_output3

    def write_results_s3(self, dataframe, output_bucket, s3_key):
        """
        Write the given dataframe results to s3.
        Args:
            dataframe: the data.
            output_bucket:  the output bucket.
            s3_key: the key

        """
        data_string = dataframe.to_csv()
        s3_object = self.s3_resource.Object(output_bucket, s3_key)  # Select folder and file name in your path in S3
        s3_object.put(Body=data_string)

    def process_data(self, event):
        """
        Process the sites.
        Args:
            event: the event containing the site list.
        """
        qa = AQ.QueryAthena(database='aggregated_metrics',
                            folder='athena-query-results/',
                            results_bucket='workspace-performance-and-insights-038954691342-us-east-1',
                            athena_client=self.athena_client,
                            s3_client=self.s3_client)

        site_set = []
        for site_json in event['Items']:
            site_set.append(site_json[SITE_ID])

        print(site_set)
        with open(self.query_file, 'r') as file:
            query = file.read().format(', '.join([f"'{site}'" for site in site_set]))
            print(query)
            qid = qa.load_conf(query)
            df_output1 = qa.get_result(qid)
            anomaly_lst = []
            extreme_lst = []
            forecast_lst= []

            threads = []
            for site in site_set:
                thread = threading.Thread(target=self.process_site, args=(anomaly_lst, df_output1, extreme_lst,forecast_lst, site))
                thread.start()
                print("Starting Thread for Site: {}", site)
                threads.append(thread)

            for thread in threads:
                thread.join()

            anomaly_lst = pd.concat(anomaly_lst)
            extreme_lst = pd.concat(extreme_lst)
            forecast_lst = pd.concat(forecast_lst)

            # Format outputs
            # merge anomaly and extreme lists
            anomaly_lst_format = anomaly_lst[
                [self.site_var, 'ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper', 'anomaly_hi', 'anomaly_lw', 'anomaly',
                 'target',
                 'ttl_vol',
                 'changepoint',
                 'delta']]
            extreme_lst_format = extreme_lst[[self.site_var, self.dt_var, self.tgt_var, VOL_VAR]]
            extreme_lst_format.reset_index()
            extreme_lst_format.columns = [self.site_var, 'ds', 'y', 'ttl_vol']
            extreme_lst_format['ds'] = extreme_lst_format['ds'].apply(pd.to_datetime).dt.date
            extreme_lst_format['target'] = self.target
            anomaly_all1 = pd.concat([anomaly_lst_format, extreme_lst_format], ignore_index=True)
            anomaly_all1['exec_dt'] = pd.to_datetime('today').date()
            anomaly_all1['target_var'] = self.tgt_var
            anomaly_all1 = anomaly_all1[anomaly_all1['ds'] >= self.start_date]

            # merge new anomaly with forecast list
            forecast_lst['target'] = self.target
            forecast_lst['target_var'] = self.tgt_var
            forecast_lst['ds'] = forecast_lst['ds'].apply(pd.to_datetime).dt.date
            forecast_lst['exec_dt'] = pd.to_datetime('today').date()
            forecast_lst.columns = ['fc','ds', 'yhat','yhat_lower','yhat_upper','ttl_vol','target','target_var','exec_dt']

            anomaly_all2= pd.concat([anomaly_all1, forecast_lst], ignore_index=True)
            print (anomaly_all2.head())

            self.write_results_s3(anomaly_all2, self.output_bucket, self.output_key)

    def process_site(self, anomaly_lst, df_output, extreme_lst, forecast_lst, site):
        """
        Process the site data.
        Args:
            anomaly_lst: the anomaly lst.
            df_output: the dataframe output.
            extreme_lst: the extreme list.
            site: the site.
        """
        df = df_output
        # call data prep function
        globals()["prophet_df1_" + str(site)], globals()["extreme_df_" + str(site)] = self.data_prep(df, site)
        # call prophet model selection function
        globals()["best_param_combo_" + str(site)] = self.prpht_para_slct(globals()["prophet_df1_" + str(site)])
        # call prophet model building function
        globals()["output_mdl_" + str(site)], globals()["output_df_" + str(site)] = self.mdl_fit(
            globals()["prophet_df1_" + str(site)], globals()["best_param_combo_" + str(site)])
        # call anomaly dataframe creation function
        anomaly_lst.append(self.format_output(globals()["output_df_" + str(site)], site))
        extreme_lst.append(globals()["extreme_df_" + str(site)])
        forecast_lst.append(self.format_out2(globals()["output_mdl_"+str(site)],df,site))
