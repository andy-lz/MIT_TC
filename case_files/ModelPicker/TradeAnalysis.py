import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import multiprocessing as mp
from scipy.stats import norm


class TradeAnalysis:

    def __init__(self, pnl_list, returns_list):
        self.pnl_list = pnl_list
        self.returns_list = returns_list

    def get_series(self, series_type):
        if series_type == "P&L":
            series = self.pnl_list
        elif series_type == "Returns":
            series = self.returns_list
        return series
    
    def calc_running_pnl_list(self, series_type):
        series = self.get_series(series_type)
        running_pnl_list = []
        counter = 0
        for profits in series:
            counter += profits
            running_pnl_list.append(counter)
        return running_pnl_list

    def plot_running_pnl(self, series_type):
        running_pnl = self.calc_running_pnl_list(series_type)
        plt.plot(running_pnl)
        plt.show()

    def calc_day_sharpe(self, series_type):
        series = self.get_series(series_type)
        mean_profits = np.mean(series)
        std_profits = np.std(series)
        sharpe = mean_profits / std_profits * np.sqrt(252)
        return sharpe
    
    def calc_trade_sharpe(self, series_type):
        series = self.get_series(series_type)
        new_series = [profit for profit in series if profit != 0]
        mean_profits = np.mean(new_series)
        std_profits = np.std(new_series)
        sharpe = mean_profits / std_profits * np.sqrt(252)
        return sharpe

    def calc_downside_deviation(self, series_type, target):
        series = self.get_series(series_type)
        num_trades = 0
        summation = 0
        for profit in series:
            summation += min(0, profit - target)**2
            num_trades += 1
        downside_deviation = np.sqrt(1.0/num_trades * summation)
        return downside_deviation

    def calc_sortino(self, series_type, target):
        series = self.get_series(series_type)
        downside_deviation = self.calc_downside_deviation(series_type, target)
        mean_profits = np.mean(series)
        if downside_deviation == 0:
            return np.nan
        sortino = mean_profits / downside_deviation * np.sqrt(252)
        return sortino

    def calc_p_value(self, series_type, target_return):
        series = self.get_series(series_type)
        mean_pnl = np.mean(series)
        std_returns = np.std(series)
        num_trades = len(series)
        z_score = (mean_pnl - target_return) / (std_returns / np.sqrt(num_trades))
        p_value = 1 - norm.cdf(z_score)
        return p_value

    def calc_num_wins(self, series_type):
        series = self.get_series(series_type)
        num_wins = 0
        for profit in series:
            if profit > 0:
                num_wins += 1
        return num_wins

    def calc_num_losses(self, series_type):
        series = self.get_series(series_type)
        num_losses = 0
        for profit in series:
            if profit < 0:
                num_losses += 1
        return num_losses

    def calc_num_trades(self, series_type):
        num_wins = self.calc_num_wins(series_type)
        num_losses = self.calc_num_losses(series_type)
        num_trades = num_wins + num_losses
        return num_trades

    def calc_win_rate(self, series_type):
        num_wins = self.calc_num_wins(series_type)
        num_losses = self.calc_num_losses(series_type)
        win_rate = num_wins / float(num_wins + num_losses)
        return win_rate

    def calc_average_win(self, series_type):
        series = self.get_series(series_type)
        positive_pnl_list = [profit for profit in series if profit > 0]
        average_win = np.mean(positive_pnl_list)
        return average_win

    def calc_average_loss(self, series_type):
        series = self.get_series(series_type)
        negative_pnl_list = [profit for profit in series if profit < 0]
        if not negative_pnl_list:
            return np.nan
        average_loss = np.mean(negative_pnl_list)
        return average_loss

    def calc_risk_reward_ratio(self, series_type):
        series = self.get_series(series_type)
        average_loss = self.calc_average_loss(series_type)
        average_win = self.calc_average_win(series_type)
        risk_reward_ratio = abs(average_loss / average_win)
        return risk_reward_ratio

    def calc_breakeven_win_rate(self, series_type):
        risk_reward_ratio = self.calc_risk_reward_ratio(series_type)
        breakeven_win_rate = risk_reward_ratio / float(risk_reward_ratio + 1)
        return breakeven_win_rate

    def calc_breakeven_risk_reward_ratio(self, series_type):
        win_rate = self.calc_win_rate(series_type)
        if win_rate == 1:
            return np.nan
        breakeven_risk_reward = win_rate / (1 - win_rate)
        return breakeven_risk_reward

    def calc_drawdown_list(self, series_type, account_size):
        running_pnl_list = self.calc_running_pnl_list("P&L")
        drawdown_list = [0]
        percent_drawdown_list = [0]
        for i in range(1, len(running_pnl_list)):
            drawdown_list.append(min(running_pnl_list[i] - max(running_pnl_list[:i+1]), drawdown_list[-1]))
        if series_type == "P&L":
            return drawdown_list
        elif series_type == "Returns":
            for i in range(1, len(running_pnl_list)):
                percent_drawdown_list.append(min(drawdown_list[i] / float(account_size + max(running_pnl_list[:i+1])), percent_drawdown_list[-1]))
            return percent_drawdown_list

    def calc_max_drawdown(self, series_type, account_size):
        drawdown_list = self.calc_drawdown_list(series_type, account_size)
        max_drawdown = drawdown_list[-1]
        return max_drawdown

    def calc_mar(self, series_type, account_size):
        series = self.get_series(series_type)
        mean_profits = np.mean(series)
        max_drawdown = self.calc_max_drawdown(series_type, account_size)
        if max_drawdown == 0:
            return np.nan
        mar = mean_profits / abs(max_drawdown) * 252
        return mar

    def calc_underwater_graph(self, series_type, account_size):
        running_pnl_list = self.calc_running_pnl_list("P&L")
        underwater_graph = []
        percent_underwater_graph = []
        for i in range(len(running_pnl_list)):
            underwater_graph.append(min(0, running_pnl_list[i] - max(running_pnl_list[:i+1])))
        if series_type == "P&L":
            return underwater_graph
        elif series_type == "Returns":
            for i in range(len(running_pnl_list)):
                percent_underwater_graph.append(min(0, underwater_graph[i] / float(account_size + max(running_pnl_list[:i+1]))))
            return percent_underwater_graph
    
    def plot_underwater_graph(self, series_type, account_size):
        underwater_graph = self.calc_underwater_graph(series_type, account_size)
        plt.plot(underwater_graph)
        plt.show()

    def create_summary_list(self, series_type, account_size):

        if series_type == "P&L" or series_type == "Returns":
            series = self.get_series(series_type)
            mean_profits = np.mean(series)
            std_profits = np.std(series)
            total_profits = np.sum(series)

            day_sharpe = self.calc_day_sharpe(series_type)
            trade_sharpe = self.calc_trade_sharpe(series_type)
            downside_deviation = self.calc_downside_deviation(series_type, 0)
            sortino = self.calc_sortino(series_type, 0)
            p_value = self.calc_p_value(series_type, 0)
            max_drawdown = self.calc_max_drawdown(series_type, account_size)
            mar = self.calc_mar(series_type, account_size)
            win_rate = self.calc_win_rate(series_type)
            breakeven_win_rate = self.calc_breakeven_win_rate(series_type)
            risk_reward_ratio = self.calc_risk_reward_ratio(series_type)
            breakeven_risk_reward_ratio = self.calc_breakeven_risk_reward_ratio(series_type)
            num_wins = self.calc_num_wins(series_type)
            num_losses = self.calc_num_losses(series_type)
            num_trades = self.calc_num_trades(series_type)
            average_win = self.calc_average_win(series_type)
            average_loss = self.calc_average_loss(series_type)
            
            if series_type == "P&L":
                summary = [("Day Sharpe Ratio", "{0:.2f}".format(day_sharpe)),
                          ("Trade Sharpe Ratio", "{0:.2f}".format(trade_sharpe)),
                          ("",""),
                          ("Mean of P&L", "$"+"{0:.2f}".format(mean_profits)),
                          ("Standard Deviation of P&L", "$"+"{0:.2f}".format(std_profits)),
                          ("",""),
                          ("Sortino Ratio", "{0:.2f}".format(sortino)),
                          ("Downside Deviation", "$"+"{0:.2f}".format(downside_deviation)),
                          ("",""),
                          ("Total P&L", "$"+"{:,}".format(int(total_profits))),
                          ("P-Value", "{0:.3f}".format(p_value)),
                          ("",""),
                          ("Maximum Drawdown", "$"+"{:,}".format(int(max_drawdown))),
                          ("MAR", "{0:.2f}".format(mar)),
                          ("",""),
                          ("Win Rate", "{0:.2f}".format(100*win_rate)+"%"),
                          ("Breakeven Win Rate", "{0:.2f}".format(100*breakeven_win_rate)+"%"),
                          ("Number of Wins", str(num_wins)),
                          ("Number of Losses", str(num_losses)),
                          ("Number of Trades", str(num_trades)),
                          ("",""),
                          ("Risk-Reward Ratio", "{0:.2f}".format(risk_reward_ratio)),
                          ("Breakeven Risk-Reward Ratio", "{0:.2f}".format(breakeven_risk_reward_ratio)),
                          ("Average Win", "$"+"{0:.2f}".format(average_win)),
                          ("Average Loss", "$"+"{0:.2f}".format(average_loss))]
                return summary
            
            elif series_type == "Returns":
                summary = [("Day Sharpe Ratio", "{0:.2f}".format(day_sharpe)),
                          ("Trade Sharpe Ratio", "{0:.2f}".format(trade_sharpe)),
                          ("",""),
                          ("Mean of Returns", "{0:.2f}".format(100*mean_profits)+"%"),
                          ("Standard Deviation of Returns", "{0:.2f}".format(100*std_profits)+"%"),
                          ("",""),
                          ("Sortino Ratio", "{0:.2f}".format(sortino)),
                          ("Downside Deviation", "{0:.2f}".format(100*downside_deviation)+"%"),
                          ("",""),
                          ("Total Returns", "{0:.2f}".format(100*total_profits)+"%"),
                          ("P-Value", "{0:.3f}".format(p_value)),
                          ("",""),
                          ("Maximum Drawdown", "{0:.2f}".format(100*max_drawdown)+"%"),
                          ("MAR", "{0:.2f}".format(mar)),
                          ("",""),
                          ("Win Rate", "{0:.2f}".format(100*win_rate)+"%"),
                          ("Breakeven Win Rate", "{0:.2f}".format(100*breakeven_win_rate)+"%"),
                          ("Number of Wins", str(num_wins)),
                          ("Number of Losses", str(num_losses)),
                          ("Number of Trades", str(num_trades)),
                          ("",""),
                          ("Risk-Reward Ratio", "{0:.2f}".format(risk_reward_ratio)),
                          ("Breakeven Risk-Reward Ratio", "{0:.2f}".format(breakeven_risk_reward_ratio)),
                          ("Average Win", "{0:.2f}".format(100*average_win)+"%"),
                          ("Average Loss", "{0:.2f}".format(100*average_loss)+"%")]
                return summary

        elif series_type == "Both":
            mean_profits = np.mean(self.returns_list)
            std_profits = np.std(self.returns_list)
            total_profits = np.sum(self.pnl_list)

            day_sharpe = self.calc_day_sharpe("Returns")
            trade_sharpe = self.calc_trade_sharpe("Returns")
            downside_deviation = self.calc_downside_deviation("Returns", 0)
            sortino = self.calc_sortino("Returns", 0)
            p_value = self.calc_p_value("Returns", 0)
            max_drawdown = self.calc_max_drawdown("Returns", account_size)
            mar = self.calc_mar("Returns", account_size)
            win_rate = self.calc_win_rate("Returns")
            breakeven_win_rate = self.calc_breakeven_win_rate("Returns")
            risk_reward_ratio = self.calc_risk_reward_ratio("Returns")
            breakeven_risk_reward_ratio = self.calc_breakeven_risk_reward_ratio("Returns")
            num_wins = self.calc_num_wins("Returns")
            num_losses = self.calc_num_losses("Returns")
            num_trades = self.calc_num_trades("Returns")
            average_win = self.calc_average_win("Returns")
            average_loss = self.calc_average_loss("Returns")

            summary = [("Day Sharpe Ratio", "{0:.2f}".format(day_sharpe)),
                      ("Trade Sharpe Ratio", "{0:.2f}".format(trade_sharpe)),
                      ("",""),
                      ("Mean of P&L", "{0:.2f}".format(100*mean_profits)+"%"),
                      ("Standard Deviation of P&L", "{0:.2f}".format(100*std_profits)+"%"),
                      ("",""),
                      ("Sortino Ratio", "{0:.2f}".format(sortino)),
                      ("Downside Deviation", "{0:.2f}".format(100*downside_deviation)+"%"),
                      ("",""),
                      ("Total P&L", "$"+"{:,}".format(int(total_profits))),
                      ("P-Value", "{0:.3f}".format(p_value)),
                      ("",""),
                      ("Maximum Drawdown", "{0:.2f}".format(100*max_drawdown)+"%"),
                      ("MAR", "{0:.2f}".format(mar)),
                      ("",""),
                      ("Win Rate", "{0:.2f}".format(100*win_rate)+"%"),
                      ("Breakeven Win Rate", "{0:.2f}".format(100*breakeven_win_rate)+"%"),
                      ("Number of Wins", str(num_wins)),
                      ("Number of Losses", str(num_losses)),
                      ("Number of Trades", str(num_trades)),
                      ("",""),
                      ("Risk-Reward Ratio", "{0:.2f}".format(risk_reward_ratio)),
                      ("Breakeven Risk-Reward Ratio", "{0:.2f}".format(breakeven_risk_reward_ratio)),
                      ("Average Win", "{0:.2f}".format(100*average_win)+"%"),
                      ("Average Loss", "{0:.2f}".format(100*average_loss)+"%")]
            return summary

    def print_summary_list(self, series_type, account_size):
        template = "{0:50} {1:20}"
        for stat in self.create_summary_list(series_type, account_size):
            print template.format(stat[0], stat[1])
            
    def get_stats(self, series_type, account_size):
        if series_type == "P&L" or series_type == "Returns":
            series = self.get_series(series_type)
            mean_profits = np.mean(series)
            std_profits = np.std(series)
            total_profits = np.sum(series)

            day_sharpe = self.calc_day_sharpe(series_type)
            trade_sharpe = self.calc_trade_sharpe(series_type)
            downside_deviation = self.calc_downside_deviation(series_type, 0)
            sortino = self.calc_sortino(series_type, 0)
            p_value = self.calc_p_value(series_type, 0)
            max_drawdown = self.calc_max_drawdown(series_type, account_size)
            mar = self.calc_mar(series_type, account_size)
            win_rate = self.calc_win_rate(series_type)
            breakeven_win_rate = self.calc_breakeven_win_rate(series_type)
            risk_reward_ratio = self.calc_risk_reward_ratio(series_type)
            breakeven_risk_reward_ratio = self.calc_breakeven_risk_reward_ratio(series_type)
            num_wins = self.calc_num_wins(series_type)
            num_losses = self.calc_num_losses(series_type)
            num_trades = self.calc_num_trades(series_type)
            average_win = self.calc_average_win(series_type)
            average_loss = self.calc_average_loss(series_type)
        
        elif series_type == "Both":
            mean_profits = np.mean(self.returns_list)
            std_profits = np.mean(self.returns_list)
            total_profits = np.sum(self.pnl_list)

            day_sharpe = self.calc_day_sharpe("Returns")
            trade_sharpe = self.calc_trade_sharpe("Returns")
            downside_deviation = self.calc_downside_deviation("Returns", 0)
            sortino = self.calc_sortino("Returns", 0)
            p_value = self.calc_p_value("Returns", 0)
            max_drawdown = self.calc_max_drawdown("Returns", account_size)
            mar = self.calc_mar("Returns", account_size)
            win_rate = self.calc_win_rate("Returns")
            breakeven_win_rate = self.calc_breakeven_win_rate("Returns")
            risk_reward_ratio = self.calc_risk_reward_ratio("Returns")
            breakeven_risk_reward_ratio = self.calc_breakeven_risk_reward_ratio("Returns")
            num_wins = self.calc_num_wins("Returns")
            num_losses = self.calc_num_losses("Returns")
            num_trades = self.calc_num_trades("Returns")
            average_win = self.calc_average_win("Returns")
            average_loss = self.calc_average_loss("Returns")
            
        stats = [p_value, average_win, average_loss, risk_reward_ratio, breakeven_risk_reward_ratio, win_rate,
                 breakeven_win_rate, num_wins, num_losses, num_trades, total_profits, max_drawdown, mar,
                 sortino, trade_sharpe, day_sharpe]
        return stats
