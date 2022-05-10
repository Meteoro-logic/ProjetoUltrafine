from datetime import datetime, date
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
import tqdm 
import matplotlib.ticker as mtick
from functools import partial 
from operator import ne 
from mpl_toolkits.mplot3d import Axes3D



# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

num_threads = 6

class MathTextSciFormatter(mtick.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

# Format with 2 decimal places
plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))


def procs(df_por_hora):
    for i in range(23):
        teste01 = [pd.DataFrame(columns=["Date", "Dp (nm)", "Estação", 'Período', 'Dia do ano', 'weekday', 'hour', "Concentração"]) for
                   i in range(23)]
        teste01[i].loc[:, "Date"] = pd.Series(np.random.randn((len(por_hora[i]) * 104)))
    for k in range(0, 2):
        b = 0
        for i in tqdm.tqdm(range(104)):
            time.sleep(0.01)
            for f in range(0, len(df_por_hora[k])):
                    teste01[k].loc[b, "Date"]= df_por_hora[k].loc[f, "Date"]
                    
                    b = b + 1
    return teste01

weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
weekend = ['Saturday', 'Sunday']
horas_dia = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
horas_noite = [19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, ]
horas_box = ["00:00","01:00","02:00","03:00","04:00","05:00","06:00",
             "07:00","08:00","09:00","10:00","11:00","12:00","13:00",
             "14:00","15:00","16:00","17:00","18:00","19:00","20:00",
             "21:00","22:00","23:00"]
Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
seasons = [('Verão', (date(Y, 1, 1), date(Y, 3, 20))),
           ('Outono', (date(Y, 3, 21), date(Y, 6, 20))),
           ('Inverno', (date(Y, 6, 21), date(Y, 9, 22))),
           ('Primavera', (date(Y, 9, 23), date(Y, 12, 20))),
           ('Verão', (date(Y, 12, 21), date(Y, 12, 31)))]
estacoes = ['Verão', 'Outono', 'Inverno', 'Primavera','Verão', 'Outono', 'Inverno', 'Primavera']

var_box = [' 10.6', ' 10.9', ' 11.3', ' 11.8', ' 12.2', ' 12.6',
           ' 13.1', ' 13.6', ' 14.1', ' 14.6', ' 15.1', ' 15.7', ' 16.3',
           ' 16.8', ' 17.5', ' 18.1', ' 18.8', ' 19.5', ' 20.2', ' 20.9',
           ' 21.7', ' 22.5', ' 23.3', ' 24.1', ' 25.0', ' 25.9', ' 26.9',
           ' 27.9', ' 28.9', ' 30.0', ' 31.1', ' 32.2', ' 33.4', ' 34.6',
           ' 35.9', ' 37.2', ' 38.5', ' 40.0', ' 41.4', ' 42.9', ' 44.5',
           ' 46.1', ' 47.8', ' 49.6', ' 51.4', ' 53.3', ' 55.2', ' 57.3',
           ' 59.4', ' 61.5', ' 63.8', ' 66.1', ' 68.5', ' 71.0', ' 73.7',
           ' 76.4', ' 79.1', ' 82.0', ' 85.1', ' 88.2', ' 91.4', ' 94.7',
           ' 98.2', '101.8', '105.5', '109.4', '113.4', '117.6', '121.9',
           '126.3', '131.0', '135.8', '140.7', '145.9', '151.2', '156.8',
           '162.5', '168.5', '174.7', '181.1', '187.7', '194.6', '201.7',
           '209.1', '216.7', '224.7', '232.9', '241.4', '250.3', '259.5',
           '269.0', '278.8', '289.0', '299.6', '310.6', '322.0', '333.8',
           '346.0', '358.7', '371.8', '385.4', '399.5', '414.2', '429.4',"Date" ]

nucleation_mode= ['10.6', ' 10.9', ' 11.3', ' 11.8', ' 12.2', ' 12.6',
                  ' 13.1', ' 13.6', ' 14.1', ' 14.6', ' 15.1', ' 15.7', 
                  ' 16.3',' 16.8', ' 17.5', ' 18.1', ' 18.8', ' 19.5', 
                  ' 20.2', ' 20.9',' 21.7', ' 22.5', ' 23.3', ' 24.1', 
                  ' 25.0']
teste01 = pd.DataFrame(columns=["Date", "Dp (nm)", "Estação", 'Período', 'Dia do ano', 'weekday', 'hour', "Concentração"])
aitken_mode= [' 25.9', ' 26.9',' 27.9', ' 28.9', ' 30.0', ' 31.1', ' 32.2',
              ' 33.4', ' 34.6',' 35.9', ' 37.2', ' 38.5', ' 40.0', ' 41.4',
              ' 42.9', ' 44.5',' 46.1', ' 47.8', ' 49.6', ' 51.4', ' 53.3', 
              ' 55.2', ' 57.3',' 59.4', ' 61.5', ' 63.8', ' 66.1', ' 68.5', 
              ' 71.0', ' 73.7',' 76.4', ' 79.1', ' 82.0', ' 85.1', ' 88.2', 
              ' 91.4', ' 94.7',' 98.2',]

Accumulation_mode =['101.8', '105.5', '109.4', '113.4', '117.6', '121.9',
           '126.3', '131.0', '135.8', '140.7', '145.9', '151.2', '156.8',
           '162.5', '168.5', '174.7', '181.1', '187.7', '194.6', '201.7',
           '209.1', '216.7', '224.7', '232.9', '241.4', '250.3', '259.5',
           '269.0', '278.8', '289.0', '299.6', '310.6', '322.0', '333.8',
           '346.0', '358.7', '371.8', '385.4', '399.5', '414.2', '429.4']



var_box2 = [' 10.6', '', '', '', '', '','', '', '', '', 
            ' 15.1', '', '','', '', '', '', '', '', '',
            ' 21.7', '', '', '', '', '', '','', '', '', 
            ' 31.1', '', '', '','', '', '', '', '', '', 
            ' 44.5','', '', '', '', '', '', '','', '', 
            ' 63.8', '','','', '','', '', '', '', '',
            ' 91.4','','','','','','', '', '','', '', 
            '135.8', '', '', '', '','', '', '', '', '', 
            '194.6', '','', '', '', '', '', '', '','', 
            '278.8', '', '', '', '', '','', '', '', '',
            '', '', '429.4', ]
def by_season_box(df1):
    lista_por_estacoes = [[] for i in range(4)]
    for i in range(0, 4):
        lista_por_estacoes[i] = df1[df1["Estação"] == estacoes[i]]
    return lista_por_estacoes

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


def by_weekday(df1_wday):
    df1__by_weekday = df1_wday[df1_wday["weekday"] <= 4]  # submatriz com somente dias da semana
    return df1__by_weekday


def by_weekend(df1_wend):
    df1_weekend_mtx = df1_wend[df1_wend["weekday"] > 4]  # submatriz com somente dias da semana
    return df1_weekend_mtx


def by_hour(df1):
    list_poe_hora = [[] for i in range(0,24)]
    for l in range(0, 24):
        list_poe_hora[l] = df1[df1["hour"] == l]
        list_poe_hora[l].reset_index(drop=True, inplace=True)
    return list_poe_hora


def by_season(df1):
    lista_por_estacoes = [[] for i in range(23)]
    for i in range(0, 4):
        x1 = by_weekday(df1)
        x2 = by_weekend(df1)
        lista_por_estacoes[i] = x1[((x1["weekday"] <= 4) & (x1["Estação"] == estacoes[i]))]
        lista_por_estacoes[i] = lista_por_estacoes[i].reset_index(drop=True)
        lista_por_estacoes[(i + 4)] = x2[((x2["weekday"] > 4) & (x2["Estação"] == estacoes[i]))]
        lista_por_estacoes[(i + 4)] = lista_por_estacoes[(i + 4)].reset_index(drop=True)
    return lista_por_estacoes


def sort_diurno_e_noturno(df1):
    global periodo_diurno, x2
    periodo_diurno = [[] for i in range(16)]
    for i in range(0, 8):
        x2 = df1
        periodo_diurno[i] = x2[i][(x2[i]["hour"] >= 7) & (x2[i]["hour"] <= 18)]
    for j in range(0, 8):
        periodo_diurno[j + 8] = x2[j][(x2[j]["hour"] >= 19) | (x2[j]["hour"] <= 6)]
    return periodo_diurno


def hour_means_values(df1):
    if type(df1) is list:
        df1_means = [[] for i in range(int(len(df1)))]
        for i in range(0, int(len(df1))):
            df1_means[i] = df1[i].groupby(df1[i]['Date'].dt.hour).mean()
    else:
        df1_means = df1.groupby(df1['Date'].dt.hour).mean()
    return df1_means


def daily_means_values(df1_daily):
    if type(df1_daily) is list:
        df1_means = [[] for i in range(int(len(df1_daily)))]
        for i3 in range(0, int(len(df1_daily))):
            df1_means[i3] = df1_daily[i3].groupby(df1_daily[i3]['Date'].dt.day).mean()
    else:
        df1_means = df1_daily.groupby(df1_daily['Date'].dt.day).mean()
    return df1_means


def monty_means_values(df1):
    if type(df1) is list:
        df1_means = [[] for i in range(int(len(df1)))]
        for i in range(0, int(len(df1))):
            df1_means[i] = df1[i].groupby(df1[i]['Date'].dt.month).mean()
    else:
        df1_means = df1.groupby(df1['Date'].dt.month).mean()
    return df1_means


def get_concen_tot(df1):
    global df_concen
    if type(df1) is list:
        df_concen = [["Date", "Total Conc.(#/cm³)", "weekday", "Estação", "hour", "Período"] for i in
                     range(int(len(df1)))]
        for i in range(0, int(len(df1))):
            df_concen[i] = df1[i][["Date", "Total Conc.(#/cm³)", "weekday", "Estação", "hour", "Período"]]
    else:
        df_concen = df1[["Date", "Total Conc.(#/cm³)", "weekday", "Estação", "hour", "Período"]]
    return df_concen


def clear_week_concen(df1_ckc):
    if type(df1_ckc) is list:
        df1_tamanhos = [[] for i in range(int(len(df1_ckc)))]
        for i in range(0, int(len(df1_ckc))):
            df1_tamanhos[i] = df1_ckc[i].drop(["Total Conc.(#/cm³)", "weekday", "hour"], axis=1)
    else:
        df1_tamanhos = df1_ckc.drop(["Total Conc.(#/cm³)", "weekday", "hour"], axis=1)
    return df1_tamanhos


def transpose_data(df1_ts):
    global new_list
    if type(df1_ts) is list:
        df1_trans = [[] for t in range(int(len(df1_ts)))]
        for o in range(0, int(len(df1_ts))):
            df1_trans[o] = np.transpose(df1_ts[o]).dropna()
            sets = [set(l) for l in df1_trans]
            new_list = [l for l, s in zip(df1_trans, sets) if not any(s < other for other in sets)]

    else:
        df1_trans = np.transpose(df1_ts).dropna()
        new_list = [i for i in df1_trans if not (i.shape[0] == 0)]
    return new_list


def plot_graf_bar_seasons_diurno(df1_4, j, horas_dia):
    par = [n for n in range(0, j * 2) if n % 2 == 0]
    impar = [n for n in range(0, j * 2) if n % 2 != 0]
    estacoes_1 = ['Verão', 'Inverno', 'Primavera', 'Verão', 'Inverno', 'Primavera']

    for k in range(0, j):
        if (k <= 2):
            fig1 = plt.figure(par[k])
            fig1.subplots_adjust(hspace=0.31, wspace=0.15)
            for w in range(1, 7):
                ax1 = fig1.add_subplot(2, 3, w)
                ax1.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_dia[w - 1]) + "h \n em dias da semana (Seg - Sex) " + estacoes_1[k], pad=15, color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax1.bar(df1_4[k].index, df1_4[k][horas_dia[w - 1]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig1.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 07 e 12h em dias da semana (Seg - Sex) " + estacoes_1[
                    k] + ".png", dpi=500)

            fig2 = plt.figure(impar[k])
            fig2.subplots_adjust(hspace=0.31, wspace=0.15)
            for w in range(1, 7):
                ax2 = fig2.add_subplot(2, 3, w)
                ax2.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_dia[w + 5]) + "h \n em dias da semana (Seg - Sex) " + estacoes_1[k], pad=15, color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=6, weight='bold')
                ax2.bar(df1_4[k].index, df1_4[k][horas_dia[w + 5]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig2.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 13 e 18h em dias da semana (Seg - Sex) " + estacoes_1[
                    k] + ".png", dpi=500)

        if k > 2:
            fig1 = plt.figure(par[k])
            fig1.subplots_adjust(hspace=0.31, wspace=0.15)
            for w in range(1, 7):
                ax1 = fig1.add_subplot(2, 3, w)
                ax1.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_dia[w - 1]) + "h \n em finais de semana (Sab e Dom) " + estacoes_1[k], pad=15, color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax1.bar(df1_4[k].index, df1_4[k][horas_dia[w - 1]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig1.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 07 e 12h em finais de semana (Sab e Dom) " + estacoes_1[
                    k] + ".png", dpi=500)

            fig2 = plt.figure(impar[k])
            fig2.subplots_adjust(hspace=0.31, wspace=0.15)
            for w in range(1, 7):
                ax2 = fig2.add_subplot(2, 3, w)
                ax2.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_dia[w + 5]) + "h \n em finais de semana (Sab e Dom) " + estacoes_1[k], pad=15, color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax2.bar(df1_4[k].index, df1_4[k][horas_dia[w + 5]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig2.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 13 e 18h em finais de semana (Sab e Dom) " + estacoes_1[
                    k] + ".png", dpi=500)


def plot_graf_bar_seasons_noturno(df1_bar_not, j, horas_noite):
    par = [n for n in range(0, j * 4) if n % 2 == 0]
    impar = [n for n in range(0, j * 4) if n % 2 != 0]
    list_estacoes = ['Verão', 'Inverno', 'Primavera', 'Verão', 'Inverno', 'Primavera', 'Verão', 'Inverno', 'Primavera',
                     'Verão', 'Inverno', 'Primavera', 'Verão', 'Inverno', 'Primavera']

    for k in range(6, j):
        if (k <= 8):
            fig1 = plt.figure(par[k])
            fig1.subplots_adjust(hspace=0.31, wspace=0.15)
            for z in range(1, 7):
                ax1 = fig1.add_subplot(2, 3, z)
                ax1.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_noite[z - 1]) + "h \n em dias da semana (Seg - Sex) " + list_estacoes[k], pad=15,
                              color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax1.bar(df1_bar_not[k].index, df1_bar_not[k][horas_noite[z - 1]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig1.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 19 e 00h em dias da semana (Seg - Sex) " +
                list_estacoes[
                    k] + ".png", dpi=500)

            fig2 = plt.figure(impar[k])
            fig2.subplots_adjust(hspace=0.31, wspace=0.15)
            for z in range(1, 7):
                ax2 = fig2.add_subplot(2, 3, z)
                ax2.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_noite[z + 5]) + "h \n em dias da semana (seg - sex) " + list_estacoes[k], pad=15,
                              color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=6, weight='bold')
                ax2.bar(df1_bar_not[k].index, df1_bar_not[k][horas_noite[z + 5]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig2.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 01 e 06h em dias da semana (Seg - Sex) " +
                list_estacoes[
                    k] + ".png", dpi=500)

        if (k > 8):
            fig1 = plt.figure(par[k])
            fig1.subplots_adjust(hspace=0.31, wspace=0.15)
            for z in range(1, 7):
                ax1 = fig1.add_subplot(2, 3, z)
                ax1.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_noite[z - 1]) + "h \n em finais de semana (Sab e Dom) " + list_estacoes[k], pad=15,
                              color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax1.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax1.bar(df1_bar_not[k].index, df1_bar_not[k][horas_noite[z - 1]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig1.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 19 e 00h em finais de semana (Sab e Dom) " +
                list_estacoes[
                    k] + ".png", dpi=500)

            fig2 = plt.figure(impar[k])
            fig2.subplots_adjust(hspace=0.31, wspace=0.15)
            for z in range(1, 7):
                ax2 = fig2.add_subplot(2, 3, z)
                ax2.set_title('Distruibuição de tamanho médio observadas as ' + str(
                    horas_noite[z + 5]) + "h \n em finais de semana (Sab e Dom) " + list_estacoes[k], pad=15,
                              color='#333333',
                              weight='bold', fontsize=10)
                plt.xticks(rotation=90, fontsize=10, weight='bold')
                ax2.xaxis.set_major_locator(plt.MaxNLocator(30))
                ax2.yaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xlim(0, 104)
                plt.ylabel("Concentração (dN#/$cm^{3}$)", fontsize=10, weight='bold')
                plt.xlabel("Tamanho(nm)", fontsize=10, weight='bold')
                ax2.bar(df1_bar_not[k].index, df1_bar_not[k][horas_noite[z + 5]],
                        width=0.8, bottom=None, align='center')
                plt.show()
            fig2.set_size_inches((30, 15), forward=False)
            plt.savefig(
                "Distruibuição de tamanho médio observadas entre 01 e 06h em finais de semana (Sab e Dom) " +
                list_estacoes[
                    k] + ".png", dpi=500)


df1 = pd.read_csv("D:\Dados\Dados Organizados SMPS\Totais.txt", sep=';', encoding='unicode_escape', header=17,
                  quotechar='"', decimal=".")


df1 = df1.drop(["Sample #", "Sample Temp (C)", "Sample Pressure (kPa)",
                'Gas Viscosity (Pa*s)', 'Mean Free Path (m)', 'Diameter Midpoint', 'Scan Up Time(s)',
                'Retrace Time(s)', 'Down Scan First', 'Scans Per Sample', 'Impactor Type(cm)',
                'Sheath Flow(lpm)', 'Aerosol Flow(lpm)', 'CPC Inlet Flow(lpm)', 'CPC Sample Flow(lpm)',
                'Low Voltage', 'High Voltage', 'Lower Size(nm)', 'Upper Size(nm)',
                'Density(g/cc)', 'Title', 'Status Flag', 'td(s)', 'tf(s)',
                'Median(nm)', 'Mean(nm)', 'Mode(nm)', 'Comment'], axis=1)
    
df1.rename(columns={'Total Conc.(#/cmÂ³)': "Total Conc.(#/cm³)"}, inplace=True)
d21 = df1["Date"]
df1["Date"] = pd.to_datetime(df1["Date"])
df1["Start Time"] = pd.to_timedelta(df1["Start Time"])
df1["Date"] = df1["Date"] + df1["Start Time"]
df1 = df1.drop("Start Time", axis=1)
df1 = df1[df1["Total Conc.(#/cm³)"] <= 40000]
df1 = df1.reset_index(drop=True)
sLength = len(df1["Date"])

df1.loc[:, "Estação"] = pd.Series(np.random.randn(sLength), index=df1.index)
df1.loc[:, "Período"] = pd.Series(np.random.randn(sLength), index=df1.index)
df1.loc[:, "Dia do ano"] = pd.Series(np.random.randn(sLength), index=df1.index)

for i in range(0, len(df1)):
    df1["Dia do ano"][i] = df1["Date"][i].timetuple().tm_yday
for i in range(0, sLength):
    x = get_season(df1["Date"][i])
    df1["Estação"][i] = x
df1["weekday"] = df1["Date"].dt.weekday
df1["hour"] = df1["Date"].dt.hour
df1_weekday = by_weekday(df1)  # submatriz com somente dias da semana
df1_weekend = by_weekend(df1)  # submatriz com somente finais de semana




df1_lista_por_dia = [[] for i in range(0, 366)]

for i in range(0, 366):
    df1_lista_por_dia[i] = df1[(df1["Dia do ano"] == i)]

df1_weekday_diurno = df1_weekday[
    (df1_weekday["hour"] >= 7) & (df1_weekday["hour"] <= 18)]  # submatriz com somente dias da semana e periodo diurno
df1_weekday_noturno = df1_weekday[
    (df1_weekday["hour"] >= 19) | (df1_weekday["hour"] <= 6)]  # submatriz com somente dias da semana e periodo noturno
df1_weekend_diurno = df1_weekend[
    (df1_weekend["hour"] >= 7) & (df1_weekend["hour"] <= 18)]  # submatriz com somente finais de semana e periodo diurno
df1_weekend_noturno = df1_weekend[(df1_weekend["hour"] >= 19) | (
        df1_weekend["hour"] <= 6)]  # submatriz com somente finais de semana e periodo noturno

#df1_weekday_diurno_medias = df1_weekday_diurno.groupby(df1_weekday_diurno['Date'].dt.hour).mean()
#df1_weekday_noturno_medias = df1_weekday_noturno.groupby(df1_weekday_noturno['Date'].dt.hour).mean()
#df1_weekend_diurno_medias = df1_weekend_diurno.groupby(df1_weekend_diurno['Date'].dt.hour).mean()
#df1_weekend_noturno_medias = df1_weekend_noturno.groupby(df1_weekend_noturno['Date'].dt.hour).mean()

#df1_weekday_diurno_medias_conc_t = df1_weekday_diurno_medias["Total Conc.(#/cm³)"]
#df1_weekday_noturno_medias_conc_t = df1_weekday_noturno_medias["Total Conc.(#/cm³)"]
#df1_weekend_diurno_medias_conc_t = df1_weekend_diurno_medias["Total Conc.(#/cm³)"]
#df1_weekend_noturno_medias_conc_t = df1_weekend_noturno_medias["Total Conc.(#/cm³)"]

#df1_weekday_diurno_medias = np.transpose(df1_weekday_diurno_medias).dropna()
#df1_weekday_noturno_medias = np.transpose(df1_weekday_noturno_medias).dropna()
#df1_weekend_diurno_medias = np.transpose(df1_weekend_diurno_medias).dropna()
#df1_weekend_noturno_medias = np.transpose(df1_weekend_noturno_medias).dropna()

por_estacoes = by_season(df1)
#dx = by_weekday(df1)
estacoes_dia_e_noite = sort_diurno_e_noturno(por_estacoes)
estacoes_dia_e_noite_valores_por_hora = hour_means_values(estacoes_dia_e_noite)
concentracao_total = get_concen_tot(df1)





for i in range(0, len(df1)):

    if df1["weekday"][i] < 5:
        df1["Período"][i] = str("Dia Útil")
    elif df1["weekday"][i] >= 5:
        df1["Período"][i] = str("Final De Semana")
df1["Período"] = df1["Período"].astype('category')


df1 = df1.drop(["Total Conc.(#/cm³)","Date","Dia do ano"], axis=1)
melted_df1 = pd.melt(df1, 
                    id_vars=["Estação","Período","weekday","hour"],# Variables to keep
                    var_name="Dp (nm)")
melted_df1["Dp (nm)"] = melted_df1["Dp (nm)"].astype('category')
por_hora = by_hour(melted_df1)

for i in range(0,23):
 por_hora[i].rename(columns={'value': "(dN/dlogDp(#/cm³)"}, inplace=True)



por_hora_melt=by_hour(melted_df1)
por_estação_box = by_season_box(melted_df1)

por_estação_box = [i for i in por_estação_box if not (i.shape[0] == 0)]
for i in range(3):
    por_estação_box[i].rename(columns={'value': "(dN/dlogDp(#/cm³)"}, inplace=True)
for i in range(0,24):
    por_hora_melt[i].rename(columns={'value': "(dN/dlogDp(#/cm³)"}, inplace=True)
    
por_estação_box_e_hora = [[] for i in range(3)]
for i in range(3):
        por_estação_box_e_hora[i] = by_hour(por_estação_box[i])

rc={'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 8, 'axes.titlesize': 8}
plt.rcParams.update(**rc)
estacoes2=['Verão', 'Inverno', 'Primavera']


por_estação_box_e_hora = [[] for i in range(3)]
for i in range(3):
        por_estação_box_e_hora[i] = by_hour(por_estação_box[i])
        
por_estação_concentracao = by_season(concentracao_total)

stat_concentracao = pd.DataFrame(columns=['Média (#/cm³)','Mediana (#/cm³)','Estação',"Período"],index=range(0,8))
  
        
for i in range(8):
    stat_concentracao["Média (#/cm³)"][i] =por_estação_concentracao[i]["Total Conc.(#/cm³)"].mean()
    stat_concentracao["Mediana (#/cm³)"][i] = por_estação_concentracao[i]["Total Conc.(#/cm³)"].median()
    stat_concentracao["Estação"][i] = estacoes[i]
for i in range(4):
    stat_concentracao["Período"][i] = str("Dia Útil")
    stat_concentracao["Período"][i+4] = str("Final De Semana")
stat_concentracao["Período"] = stat_concentracao["Período"].astype('category')
    
stat_concentracao = stat_concentracao.dropna()    

df1_weekday_estacoes=by_season(df1_weekday)
df1_weekend_estacoes=by_season(df1_weekend)


df1_weekday_estacoes= [x for x in df1_weekday_estacoes if len(x) > 0]
df1_weekday_estacoes_hour = [[] for i in range(3)]

df1_weekend_estacoes= [x for x in df1_weekend_estacoes if len(x) > 0]
df1_weekend_estacoes_hour = [[] for i in range(3)]

melted_df1_weekday_estacoes_hour = [[] for i in range(3)]
melted_df1_weekend_estacoes_hour = [[] for i in range(3)]


for i in range(3):
    df1_weekend_estacoes_hour[i]=hour_means_values(df1_weekend_estacoes[i])
    df1_weekend_estacoes_hour[i] = df1_weekend_estacoes_hour[i].drop([ 'Geo. Mean(nm)',"Período","weekday","Total Conc.(#/cm³)", "Dia do ano"], axis=1)
    melted_df1_weekend_estacoes_hour[i]= pd.melt( df1_weekend_estacoes_hour[i], 
                    id_vars=["hour"],# Variables to keep
                    var_name="Dp (nm)")   
    melted_df1_weekend_estacoes_hour[i]["Dp (nm)"] = melted_df1_weekend_estacoes_hour[i]["Dp (nm)"].astype(float)
    df1_weekend_estacoes_hour[i]['hour']=pd.Categorical(df1_weekend_estacoes_hour[i]['hour'])
    df1_weekend_estacoes_hour[i]['hour']=df1_weekend_estacoes_hour[i]['hour'].cat.codes
    
for i in range(3):
    df1_weekday_estacoes_hour[i]=hour_means_values(df1_weekday_estacoes[i])
    df1_weekday_estacoes_hour[i] = df1_weekday_estacoes_hour[i].drop([ 'Geo. Mean(nm)',"Período","weekday","Total Conc.(#/cm³)", "Dia do ano"], axis=1)
    melted_df1_weekday_estacoes_hour[i]= pd.melt( df1_weekday_estacoes_hour[i], 
                    id_vars=["hour"],# Variables to keep
                    var_name="Dp (nm)")   
    melted_df1_weekday_estacoes_hour[i]["Dp (nm)"] = melted_df1_weekday_estacoes_hour[i]["Dp (nm)"].astype(float)
    df1_weekday_estacoes_hour[i]['hour']=pd.Categorical(df1_weekday_estacoes_hour[i]['hour'])
    df1_weekday_estacoes_hour[i]['hour']=df1_weekday_estacoes_hour[i]['hour'].cat.codes  
        
        

    
for i in range(3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(melted_df1_weekday_estacoes_hour[i]["hour"], melted_df1_weekday_estacoes_hour[i]["Dp (nm)"], melted_df1_weekday_estacoes_hour[i]["value"], cmap=plt.cm.inferno, linewidth=0.2,shade=True)
# to Add a color bar which maps values to colors.
    
    surf=ax.plot_trisurf(melted_df1_weekday_estacoes_hour[i]["hour"], melted_df1_weekday_estacoes_hour[i]["Dp (nm)"], melted_df1_weekday_estacoes_hour[i]["value"], cmap=plt.cm.inferno,shade=False, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=7,pad = -0.08,orientation="vertical")
# Rotate it
    ax.view_init(47, 23)
# Other palette
    ax.plot_trisurf(melted_df1_weekday_estacoes_hour[i]["hour"], melted_df1_weekday_estacoes_hour[i]["Dp (nm)"], melted_df1_weekday_estacoes_hour[i]["value"], cmap=plt.cm.inferno, linewidth=0.02)
    ax.set_xlabel('Hora (hh)',fontsize=12,y=1.03,weight='bold')
    ax.set_ylabel("Dp (nm)",fontsize=12,y=1.03,weight='bold')
    ax.set_zlabel("Concentração (dN/dlogDp(#/cm³)",fontsize=12,weight='bold')
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    ax.zaxis.labelpad=10
    plt.xticks(range(1, 24, 3),fontsize=12)
    plt.yticks(range(0, 450, 50),fontsize=12)
    plt.gca().zaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    plt.title('Distribuição do tamanho médio de partículas por hora durante semana no/na '+str(estacoes2[i]) ,fontsize=12,y=1.03,weight='bold')
    fig.set_size_inches((20, 10), forward=False)
    plt.savefig('distribuição das ' + str(estacoes2[i]) + ' horas ' '.png',dpi=100)
    plt.show()
    
for i in range(3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(melted_df1_weekend_estacoes_hour[i]["hour"], melted_df1_weekend_estacoes_hour[i]["Dp (nm)"], melted_df1_weekend_estacoes_hour[i]["value"], cmap=plt.cm.inferno, linewidth=0.2,shade=True)
# to Add a color bar which maps values to colors.
    surf=ax.plot_trisurf(melted_df1_weekend_estacoes_hour[i]["hour"], melted_df1_weekend_estacoes_hour[i]["Dp (nm)"], melted_df1_weekend_estacoes_hour[i]["value"], cmap=plt.cm.inferno,shade=False, linewidth=0.2)
    fig.colorbar( surf, shrink=0.5, aspect=7,pad = -0.08,orientation="vertical")
# Rotate it
    ax.view_init(47, 23)
# Other palette
    ax.plot_trisurf(melted_df1_weekend_estacoes_hour[i]["hour"], melted_df1_weekend_estacoes_hour[i]["Dp (nm)"], melted_df1_weekend_estacoes_hour[i]["value"], cmap=plt.cm.inferno, linewidth=0.02)
    ax.set_xlabel('Hora (hh)',fontsize=12,y=1.03,weight='bold')
    ax.set_ylabel("Dp (nm)",fontsize=12,y=1.03,weight='bold')
    ax.set_zlabel("Concentração (dN/dlogDp(#/cm³)",fontsize=12,weight='bold')
    ax.xaxis.labelpad=10
    ax.yaxis.labelpad=10
    ax.zaxis.labelpad=10
    plt.xticks(range(1, 12, 3),fontsize=12)
    plt.yticks(range(0, 450, 50),fontsize=12)
    plt.gca().zaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
    plt.title('Distribuição do tamanho médio de partículas por hora aos finais de semana durante o/a '+str(estacoes2[i]) ,fontsize=12,y=1.03,weight='bold')
    fig.set_size_inches((20, 10), forward=False)
    plt.savefig('distribuição das ' + str(estacoes2[i]) + ' horas ' '.png',dpi=100)
    plt.show()



for j in range(3):
 for i in range(0,24):
     
    ax =sns.catplot(x="Dp (nm)", y="(dN/dlogDp(#/cm³)",
            col='Período', aspect=1,
            kind="box", data=por_estação_box_e_hora[j][i],showfliers = False ) #flierprops = dict(markerfacecolor = '0.50', markersize = 2)#
    plt.subplots_adjust(top=0.9)
    ax.fig.suptitle('Boxplot das concentrações as '+str(horas_box[i])+" horas  "+str(estacoes2[j])+".", fontsize=8)
    ax.set_xticklabels(var_box2, rotation='vertical', fontsize=8)
    
   
    
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.e"))
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    x=str(horas_box[i])
    
    plt.savefig('Boxplot das ' + str(i) + ' hora '  + str(estacoes2[j]) + '.png',dpi=200)




for i in range (0,24):
    ax=sns.catplot(x="Dp (nm)", y="(dN/dlogDp(#/cm³)", col="Estação",row="Período", data=por_hora_melt[i], kind = "box",showfliers = True) 
    
    plt.subplots_adjust(top=0.9)
    print(i)
    
    ax.set_xticklabels(var_box2, rotation='vertical', fontsize=8)
    
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))  
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    
    plt.suptitle('Boxplot das concentrações as '+str(horas_box[i])+" horas  "".", fontsize=12,y=1.03,weight='bold')
    
    plt.ylim(0,60000)
  
    
    x=str(horas_box[i])

    plt.savefig('Boxplot das ' + str(i) + ' horas ' '.png',dpi=200,bbox_inches = "tight")
    
for i in range(3):
        
    melted_df1_weekend_estacoes_hour[i].to_csv(r"D:\Python Scripts\weekend"+str(estacoes2[i])+".csv")
    melted_df1_weekday_estacoes_hour[i].to_csv(r"D:\Python Scripts\weekday"+str(estacoes2[i])+".csv")


matriz_final.to_csv(r"D:\Python Scripts\matriz_final.csv")


matriz_final.rename(columns={'Data': "Date",'Concentração de CO (Monóxido de Carbono) em ppmMarg.Tietê-Pte Remédios':"Monóxido de Carbono",
                             'Concentração de MP2.5 (Partículas Inaláveis Finas) em µg/m3Cid.Universitária-USP-Ipen':"MP2.5",
                             'Concentração de NO2 (Dióxido de Nitrogênio) em µg/m3Marg.Tietê-Pte Remédios':"Dióxido de Nitrogênio",
                             'Concentração de O3 (Ozônio) em µg/m3Cid.Universitária-USP-Ipen':"Ozônio",
                             'Concentração de TEMP (Temperatura do Ar) em °CMarg.Tietê-Pte Remédios':"Temperatura do Ar",
                             'Concentração de UR (Umidade Relativa do Ar) em %Marg.Tietê-Pte Remédios':"Umidade Relativa",
                             }, inplace=True)
matriz_final["Monóxido de Carbono"
                             ].astype(str).astype(float)    
matriz_final["Ozônio"
                             ].astype(str).astype(int)
matriz_final["Dióxido de Nitrogênio"
                             ].astype(str).astype(float)
matriz_final["Temperatura do Ar"
                             ].astype(str).astype(float)
matriz_final["Umidade Relativa"
                             ].astype(str).astype(float)

concentracao_total.to_csv(r"D:\Python Scripts\concentra.csv")
df1_weekday_estacoes_2= hour_means_values(df1_weekday_estacoes)
df1_weekend_estacoes_2= hour_means_values(df1_weekend_estacoes)


for i in range(3):
        
    df1_weekday_estacoes_2[i].to_csv(r"D:\Python Scripts\weekend_geo"+str(estacoes2[i])+".csv")
    df1_weekend_estacoes_2[i].to_csv(r"D:\Python Scripts\weekday_geo"+str(estacoes2[i])+".csv")


    
plt.show()
