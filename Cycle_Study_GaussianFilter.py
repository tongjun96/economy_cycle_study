import numpy as np
from scipy import signal, interpolate
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as  plt
import matplotlib as mpl
import calendar, datetime

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取上证综指历史数据，这里我使用的是Tushare,如果有的话可以去申请一个，没有也可以用自己的数据，把下面这段改一下
import tushare as ts

ts.set_token('9b5781c35d09ed01f8c24414a0196e778d8e52e727b6f4bece2e494b')
pro = ts.pro_api()
shanghai = pro.index_monthly(ts_code='000001.SH', start_date='19950101', end_date='20190831', \
                             fields='trade_date,close').sort_index(ascending=False)
shanghai = shanghai.rename(columns={'close': '上证综指'})
a_seq = np.array(shanghai['上证综指'])

# 求对数同序列
log_a_seq = np.log(a_seq[12:]) - np.log(a_seq[:-12])  # 对数同比序列
data = shanghai[12:].copy()
data['同比'] = log_a_seq


def interpolation(data):
    '''
    :param data:
    :return: 对data立面的NaN值插值处理
    '''
    index_data = list(range(len(data)))
    pd_a = pd.Series(~np.isnan(data))  # 由True和False组成的序列，True表明是非NaN值
    ind_a = pd_a[pd_a].index.tolist()  # 找出非NaN的索引值
    pd_nan = pd.Series(np.isnan(data))
    ind_nan = pd_nan[pd_nan].index.tolist()  # 找出NaN的索引值
    y_data = data[ind_a]  # 非NaN值的序列
    y1_data = interpolate.interp1d(ind_a, y_data, kind='cubic')  # 立方插值
    y2_data = data
    for i in index_data:
        if i in ind_nan:
            y2_data[i] = y1_data(i)
    return y2_data


def period_mean_fft(data, nfft, peak_num, figure_flag):
    '''
    :param data:对数同比序列
    :param nfft:补零后长度
    :param peak_num:要找几个峰值
    :param figure_flag:是否画频谱图
    :return:傅里叶变换后的序列
    '''
    data_n = np.array(data)
    if len(data_n.shape) > 1 and data_n.shape[-1] > data_n.shape[0]:
        data_n = data_n.T
    Y_fft = np.fft.fft(data_n, nfft)
    nfft = len(Y_fft)
    Y_fft = np.delete(Y_fft, 0)
    power = np.abs(np.square(Y_fft[:int(np.floor(nfft / 2))]))  # 求功率谱
    amplitude = np.abs(Y_fft[:int(np.floor(nfft / 2))])  # 求振幅
    nyquist = 1 / 2
    freq = [nyquist * i / np.floor(nfft / 2) for i in range(1, int(np.floor(nfft / 2) + 1))]
    freq = np.array(freq)
    period = 1 / freq
    loc_raw = list(signal.find_peaks(power)[0])
    peak_raw = power[loc_raw]
    data_dict = {key: value for key, value in zip(loc_raw, peak_raw)}
    temp = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
    loc_peaks = temp[:peak_num]
    posi = [loc_peaks[i][0] for i in range(peak_num)]
    T_fft = period[posi]
    if figure_flag == 1:
        plt.figure(figsize=(20, 10), dpi=200)
        plt.plot(period, power)
        plt.grid(True)
        plt.xlabel('周期(月)')
        plt.ylabel('振幅')
        plt.title('周期-振幅图')
        for i in range(peak_num):
            plt.text(period[loc_peaks[i][0]], loc_peaks[i][1],
                     '[' + str(round(period[loc_peaks[i][0]], 2)) + ',' + str(round(loc_peaks[i][1])) + ']')
        plt.xlim(12, 300)
        plt.savefig('周期振幅图')
    return T_fft


def gauss_wave_predict(wave, period, n_fft, n_predict, gauss_alpha):
    '''
    高斯滤波
    '''
    # 1、填充0
    wave_pad = np.concatenate([np.zeros(n_fft - len(wave), ), wave])
    # 2、进行FFT变换
    wave_fft = np.fft.fft(wave_pad, n_fft)
    # 3、生成高斯滤波频率响应，注意这里只刻画了低频部分，后续做共轭对称处理
    gauss_index = [i for i in range(int(n_fft))]
    center_frequency = n_fft / period + 1
    gauss_win = np.exp(-np.power((np.array(gauss_index) - center_frequency), 2) / (gauss_alpha ** 2))
    # 4、频域滤波，因为时域为实数，所以频域序列有共轭对称的属性
    wave_filter = np.multiply(wave_fft, gauss_win)
    if np.mod(n_fft, 2) == 0:
        wave_filter[int(n_fft / 2 + 1):] = np.conj(wave_filter[int(n_fft / 2 - 1):0:-1])
    else:
        wave_filter[int((n_fft - 1) / 2 + 1):] = np.conj(wave_filter[int((n_fft - 1) / 2):0:-1])

    # 5、逆傅里叶变换得到时域还原序列，外延预测本质上是在延拓主值序列
    ret = np.fft.ifft(wave_filter).real
    output = np.concatenate([ret[int(len(ret) - len(wave)):], ret[:int(n_predict)]])

    return output


def regress_predict_output_f(seq, predict_len, pad_to_len, gauss_alpha, mean_flag, period_flag, peak_num, figure_flag):
    '''
    回归、预测、输出
    '''
    if mean_flag == 0:
        seq_len = len(seq)  # 输入序列的长度
        trend_a_seq = np.zeros((seq_len, 1))  # 输出去除方式，不处理的意思即为全0序列
        d_a_seq = seq
        predict_trend_seq = np.zeros((seq_len + predict_len, 1))  # 输出序列长度为输入序列长度+预测长度，因为没处理原数据，基准值为0
    elif mean_flag == 1:
        seq_len = len(seq)
        trend_a_seq = np.nanmean(seq)  # 均值，常量
        d_a_seq = seq - trend_a_seq
        predict_trend_seq = np.ones((seq_len + predict_len, 1)) * np.nanmean(seq)  # 基准值为均值
    else:
        seq_len = len(seq)
        d_a_seq = signal.detrend(seq)  # 去趋势后的序列
        trend_a_seq = seq - d_a_seq  # 趋势值，长度为trend_a_seq的序列
        predict_trend_seq = np.ones((seq_len + predict_len)) * np.nan
        predict_trend_seq[:seq_len] = trend_a_seq  # 输出序列原数据长度部分为 趋势值序列
        predict_trend_seq = interpolation(predict_trend_seq)  # 输出序列预测长度部分 为用趋势值序列插值的结果
    if period_flag == '固定周期':
        period = [42, 70, 200]
    else:
        period = period_mean_fft(d_a_seq, pad_to_len, peak_num, figure_flag)
        if len(period) == 2:
            period.append(200)
        elif len(period) == 1:
            if period[0] < 60:
                period.extend([100, 200])
            else:
                period.extend([40, 200])
    # 在对数同比序列前面补0，提升频域分辨率
    d_seq_pad = np.zeros((pad_to_len,))
    d_seq_pad[-seq_len:] = d_a_seq
    # 高斯滤波获取三周期对应的序列以及预测结果
    filter_result = np.zeros((pad_to_len + predict_len, len(period)))
    for iPeriod in range(len(period)):
        filter_result[:, iPeriod] = gauss_wave_predict(d_seq_pad, period[iPeriod], pad_to_len, predict_len, gauss_alpha)
    filter_result = filter_result[-(seq_len + predict_len):, :]

    Y = pd.DataFrame(d_a_seq)
    regress_result = np.zeros((4, 6))
    # 单变量回归
    for iPeriod in range(len(period)):
        X = pd.DataFrame(filter_result[:seq_len, iPeriod])
        X = sm.add_constant(X)
        est = sm.OLS(Y, X).fit()
        regress_result[iPeriod, :2] = np.array(est.params)
        regress_result[iPeriod, 4] = est.rsquared
        regress_result[iPeriod, 5] = est.f_pvalue
    # 多变量回归
    X = pd.DataFrame(filter_result[:seq_len, :])
    X = sm.add_constant(X)
    est = sm.OLS(Y, X).fit()
    regress_result[iPeriod, :4] = np.array(est.params)
    regress_result[iPeriod, 4] = est.rsquared
    regress_result[iPeriod, 5] = est.f_pvalue
    # pdb.set_trace()
    predict_result_temp = np.dot(np.concatenate([np.ones((seq_len + predict_len, 1)), filter_result], axis=1),
                                 np.array(est.params))
    predict_result = predict_result_temp + predict_trend_seq
    # pdb.set_trace()
    return d_a_seq, trend_a_seq, filter_result, predict_trend_seq, predict_result_temp, predict_result, period, regress_result


predict_len = 12 * 5  # 预测长度，单位为月
pad_to_len = 4096  # 填0后长度，填0是为了提升频谱分辨率
gauss_alpha = 1  # 高斯滤波器带宽，推荐设置为1
mean_flag = 1  # 数据处理方式1：去均值项  0：不处理  2：去趋势项
period_flag = '变化周期'  # 中心频率选择方式：固定周期取42，100，200 ；变化周期由傅里叶变换则计算得出
peak_num = 3  # 提取三大周期。
figure_flag = 0  # 0：傅里叶变换不画图 1：画图

[d_a_seq, trend_a_seq, filter_result, predict_trend_seq, predict_result_temp, predict_result, period, regress_result] \
    = regress_predict_output_f(log_a_seq[0:], predict_len, pad_to_len, gauss_alpha, mean_flag, period_flag, peak_num,
                               figure_flag)

pre_result = predict_result_temp + predict_trend_seq.reshape(len(predict_trend_seq))

date = [datetime.datetime.strptime(x, '%Y%m%d') for x in list(data['trade_date'])]
max_months = max(len(log_a_seq), len(pre_result))
while (True):
    if len(date) < max_months:
        if date[-1].month != 12:
            year = date[-1].year
            month = date[-1].month
        else:
            year = date[-1].year + 1
            month = 1
        date.append(date[-1] + datetime.timedelta(days=calendar.monthrange(year, month)[1]))
    else:
        break
date = [str(x)[:4] + str(x)[5:7] for x in date]

fig = plt.figure(figsize=(16, 8), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
l1, = ax.plot(range(0, len(log_a_seq)), log_a_seq, 'b', label='同比序列')
l3, = ax.plot(range(0, len(pre_result)), pre_result, 'y', label='预测走势')
xticks = [x for x in
          range(0, (datetime.date.today().year - datetime.datetime.strptime(date[0], '%Y%m').year) * 12 + 6, 12)]
xticks.extend([x for x in
               range((datetime.date.today().year - datetime.datetime.strptime(date[0], '%Y%m').year) * 12 + 6,
                     len(date), 6)])
date_display = date[0:(datetime.date.today().year - datetime.datetime.strptime(date[0], '%Y%m').year) * 12 + 6:12]
date_display.extend(date[(datetime.date.today().year - datetime.datetime.strptime(date[0], '%Y%m').year) * 12 + 6::6])
ax.set_xticks(xticks)
ax.set_xticklabels(date_display, rotation=30)
ax2 = ax.twinx()
l2, = ax2.plot(range(0, len(log_a_seq)), data['上证综指'], 'r', label='上证综指')
# ax.grid(True)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles=[l1, l2, l3], loc='upper center', fontsize=20, frameon=False, ncol=2, columnspacing=10)
plt.savefig('预测', bbox_inches='tight')