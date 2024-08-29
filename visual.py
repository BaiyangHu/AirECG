'''
Author: Jingcheng Zhao
Update: 2024-08-27
Description: 用于可视化mmwave与ECG数据的匹配
适配3分钟数据
修复bug：ECG数据长度不匹配
'''

import sys
import json
import numpy as np
import pickle
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QLineEdit, QLabel, QGridLayout, QMessageBox
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QObject, pyqtSlot, QUrl, QRect
from PyQt5.QtGui import QFont
import matplotlib
import process
import scipy.interpolate

class DataBridge(QObject):
    def __init__(self, parent=None):
        super(DataBridge, self).__init__(parent)
        self.app = parent

    @pyqtSlot(int)
    def handle_point_clicked(self, axis):
        # 转换成体素坐标
        x = axis // (9 * 17)
        y = (axis % (9 * 17)) // 17
        z = (axis % (9 * 17)) % 17
        print(f"Point clicked at: ({x}, {y}, {z})")
        self.app.plot_phase((x, y, z))

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=False)  # 仅启用x轴的鼠标缩放

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            axis = 0
        super().wheelEvent(ev, axis=axis)  # 仅对x轴进行缩放

class VisualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Visualization")
        self.setGeometry(0, 0, 1200, 800)

        # 主窗口设置
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # 设置布局
        self.main_widget.setGeometry(QRect(0, 0, 1200, 800))

        # 3D图表视图
        self.browser = QWebEngineView(self.main_widget)
        self.browser.setGeometry(QRect(20, 20, 300, 300))

        # 文本框与按钮
        self.gui_pkl_path_input = QLineEdit(self.main_widget) # pkl路径输入框
        self.gui_pkl_path_input.setGeometry(QRect(340, 20, 380, 20))
        self.gui_ecg_path_input = QLineEdit(self.main_widget) # ecg路径输入框
        self.gui_ecg_path_input.setGeometry(QRect(340, 50, 380, 20))

        self.gui_pkl_start_time = QLineEdit(self.main_widget) # pkl起始时间
        self.gui_pkl_start_time.setGeometry(QRect(440, 80, 120, 20))
        self.gui_pkl_end_time = QLineEdit(self.main_widget) # pkl结束时间
        self.gui_pkl_end_time.setGeometry(QRect(440, 110, 120, 20))
        self.gui_pkl_duration = QLineEdit(self.main_widget) # pkl持续时间
        self.gui_pkl_duration.setGeometry(QRect(440, 140, 120, 20))
        self.gui_ecg_start_time = QLineEdit(self.main_widget) # ecg起始时间
        self.gui_ecg_start_time.setGeometry(QRect(670, 80, 120, 20))
        self.gui_ecg_end_time = QLineEdit(self.main_widget) # ecg结束时间
        self.gui_ecg_end_time.setGeometry(QRect(670, 110, 120, 20))
        self.gui_ecg_duration = QLineEdit(self.main_widget) # ecg持续时间
        self.gui_ecg_duration.setGeometry(QRect(670, 140, 120, 20))

        self.gui_pkl_path_button = QPushButton('pkl', self.main_widget) # pkl导入按钮
        self.gui_pkl_path_button.setGeometry(QRect(730, 20, 50, 20))
        self.gui_pkl_path_button.clicked.connect(self.load_pkl_file)

        self.gui_ecg_path_button = QPushButton('ecg', self.main_widget) # ecg导入按钮
        self.gui_ecg_path_button.setGeometry(QRect(730, 50, 50, 20))
        self.gui_ecg_path_button.clicked.connect(self.load_ecg_file)

        # self.gui_matching_button = QPushButton('match', self.main_widget) # 开始匹配按钮
        # self.gui_matching_button.setGeometry(QRect(20, 660, 50, 20))
        # self.gui_matching_button.clicked.connect(self.match_mode)


        self.gui_output_button = QPushButton('output', self.main_widget) # 导出按钮
        self.gui_output_button.setGeometry(QRect(80, 660, 50, 20))
        self.gui_output_button.clicked.connect(self.output)

        self.gui_mmwave_start_time_label = QLabel('PKL Start', self.main_widget)
        self.gui_mmwave_start_time_label.setGeometry(QRect(340, 80, 600, 20))
        self.gui_mmwave_end_time_label = QLabel('PKL End', self.main_widget)
        self.gui_mmwave_end_time_label.setGeometry(QRect(340, 110, 600, 20))
        self.gui_mmwave_duration_label = QLabel('PKL Duration', self.main_widget)
        self.gui_mmwave_duration_label.setGeometry(QRect(340, 140, 600, 20))
        self.gui_ecg_start_time_label = QLabel('ECG Start', self.main_widget)
        self.gui_ecg_start_time_label.setGeometry(QRect(570, 80, 600, 20))
        self.gui_ecg_end_time_label = QLabel('ECG End', self.main_widget)
        self.gui_ecg_end_time_label.setGeometry(QRect(570, 110, 600, 20))
        self.gui_ecg_duration_label = QLabel('ECG Duration', self.main_widget)
        self.gui_ecg_duration_label.setGeometry(QRect(570, 140, 600, 20))

        # Phase 图
        self.phase_plot = pg.PlotWidget(self.main_widget, title="Phase")
        self.phase_plot.setGeometry(QRect(340, 170, 440, 150))

        # ECG 与二阶导数图
        self.match_plot = pg.GraphicsLayoutWidget(self.main_widget)
        self.ecg_plot = self.match_plot.addPlot(title="ECG", row=1, col=0)
        self.second_derivative_plot = self.match_plot.addPlot(title="Second Derivative", row=2, col=0)
        self.match_plot.setGeometry(QRect(20, 340, 760, 300))

        # 用于微调的plot
        self.adjust_plot = pg.PlotWidget(self.main_widget, title="Adjust")
        self.adjust_plot.setGeometry(QRect(20, 700, 760, 300))
        self.adjust_plot.setMouseEnabled(x=True, y=False)  # 仅启用x轴的鼠标缩放

        # 设置WebChannel
        self.channel = QWebChannel()
        self.bridge = DataBridge(self)
        self.channel.registerObject('bridge', self.bridge)
        self.browser.page().setWebChannel(self.channel)

    def load_pkl_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "导入 Pkl 文件", "", "Pkl Files (*.pkl)")
        if file_name:
            self.gui_pkl_path_input.setText(file_name)
            self.load_mmwave(file_name)

    def load_ecg_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "导入 ECG 文件", "", "CSV Files (*.csv)")
        if file_name:
            self.gui_ecg_path_input.setText(file_name)
            self.load_ecg(file_name)

    def load_mmwave(self, path):
        '''
        读取 mmwave 数据（pkl格式），更新起始时间值，计算结束时间，绘制3D图表
        '''
        with open(path, "rb") as f:
            d_1 = pickle.load(f)
        info = d_1[1]
        print(info)
        self.data_array = np.array(d_1[0])

        # 计算时间
        ## 获取时间戳
        self.mmwave_start_time = float(info['start_time'].replace('\n','')) # 新的beamforming会保存start_time
        self.mmwave_end_time = float(self.mmwave_start_time) + (1/params["sampling_rate"]) * (self.data_array.shape[3])
        self.mmwave_duration = float(self.mmwave_end_time) - float(self.mmwave_start_time)
        print('Start Time:', self.mmwave_start_time)
        # 时间序列
        self.mmwave_time = np.linspace(0, self.mmwave_duration, self.data_array.shape[3]+1)
        self.mmwave_time = self.mmwave_time[:-1]
        self.gui_pkl_start_time.setText(str(self.mmwave_start_time))
        self.gui_pkl_end_time.setText(str(self.mmwave_end_time))
        self.gui_pkl_duration.setText(str(self.mmwave_duration))

        sum_power = np.sum(np.abs(self.data_array) ** 2, axis=3) / self.data_array.shape[3]
        normalized_power = (sum_power.flatten() - np.min(sum_power)) / (np.max(sum_power) - np.min(sum_power))

        cmap = matplotlib.cm.get_cmap('viridis')
        colors = cmap(normalized_power)  # 将归一化的功率值映射到颜色映射
        colors = np.column_stack((colors[:, :3] * 255, colors[:, 3]))  # 转换为RGB格式

        x = np.linspace(info['voxel_range']['x'][0], info['voxel_range']['x'][1], info['voxel_grid_dimensions'][0])
        y = np.linspace(info['voxel_range']['y'][0], info['voxel_range']['y'][1], info['voxel_grid_dimensions'][1])
        z = np.linspace(info['voxel_range']['z'][0], info['voxel_range']['z'][1], info['voxel_grid_dimensions'][2])
        x, y, z = np.meshgrid(x, y, z, indexing='ij')

        pos = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T.tolist()  # 体素位置转换为列表
        colors = colors.tolist()  # 颜色转换为列表

        # 调试信息
        print("Pos shape:", np.array(pos).shape)
        print("Colors shape:", np.array(colors).shape)

        # 加载ECharts图表
        self.load_chart(pos, colors)

    def load_chart(self, pos, colors):
        # ECharts的HTML内容
        html_content = f"""
        <!DOCTYPE html>
        <html style="height: 100%">
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.0.2/dist/echarts.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/echarts-gl@2.0.9/dist/echarts-gl.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/qwebchannel@6.2.0/qwebchannel.js"></script>
        </head>
        <body style="height: 100%; margin: 0">
            <div id="main" style="height: 100%"></div>
            <script type="text/javascript">
                new QWebChannel(qt.webChannelTransport, function(channel) {{
                    window.bridge = channel.objects.bridge;
                }});
                
                var chart = echarts.init(document.getElementById('main'));
                var option = {{
                    tooltip: {{}},
                    xAxis3D: {{
                        type: 'value'
                    }},
                    yAxis3D: {{
                        type: 'value'
                    }},
                    zAxis3D: {{
                        type: 'value'
                    }},
                    grid3D: {{
                        viewControl: {{
                            projection: 'perspective'
                        }}
                    }},
                    series: [{{
                        type: 'scatter3D',
                        data: {json.dumps(pos)},
                        symbolSize: 8,
                        itemStyle: {{
                            opacity: 0.8,
                            color: function (params) {{
                                return 'rgba(' + {json.dumps(colors)}[params.dataIndex] + ')';
                            }}
                        }}
                    }}]
                }};
                chart.setOption(option);

                chart.on('click', function (params) {{
                    if (params.componentType === 'series') {{
                        // alert('Point clicked at: ' + params.value);
                        // alert('Data index: ' + params.dataIndex);
                        var dataIndex = params.dataIndex;
                        var data = {json.dumps(pos)}[dataIndex];
                        window.bridge.handle_point_clicked(dataIndex);
                    }}
                }});
            </script>
        </body>
        </html>
        """

        # 将HTML内容加载到QWebEngineView中
        self.browser.setHtml(html_content)

    def interpolate_ecg(self):

        # 相对时间
        ecg_relative_time = self.ecg_data[:, 0] - self.mmwave_start_time

        ecg_interp = scipy.interpolate.interp1d(
            ecg_relative_time,
            self.ecg_data[:, 1],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
            )
        
        # 新的时间轴
        interval = 1 / params["sampling_rate"]
        new_time_axis_start = self.ecg_start_time - self.mmwave_start_time
        new_time_axis_end = self.ecg_end_time - self.mmwave_start_time
        new_time_axis = np.arange(new_time_axis_start, new_time_axis_end + interval, interval)

        resampled_ecg = ecg_interp(new_time_axis)


        return np.column_stack((new_time_axis, resampled_ecg)) 
    
    def load_ecg(self, path):
        # 读取ECG数据
        self.ecg_data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0, 1))
        self.ecg_data[:, 0] = self.ecg_data[:, 0] / 1000

        # 计算时间
        self.ecg_start_time = int(self.ecg_data[0][0]) + 1
        self.ecg_end_time = int(self.ecg_data[-1][0])
        self.ecg_duration = self.ecg_end_time - self.ecg_start_time

        # 更新文本框
        self.gui_ecg_start_time.setText(str(self.ecg_start_time))
        self.gui_ecg_end_time.setText(str(self.ecg_end_time))
        self.gui_ecg_duration.setText(str(self.ecg_duration))

        # 截取时间段
        index = np.where((self.ecg_data[:, 0] >= self.ecg_start_time) & (self.ecg_data[:, 0] < self.ecg_end_time))
        start_index = index[0][0]
        end_index = index[0][-1]

        print('Original Start Index:', start_index)
        print('Original End Index:', end_index)
        print('Original Shape:', self.ecg_data.shape)

        # 确保我们有足够的数据点
        if (end_index - start_index) < (self.ecg_duration - 1) * 500:
            self.ecg_duration -= 1
            self.ecg_end_time -= 1
            end_index = np.where(self.ecg_data[:, 0] < self.ecg_end_time)[0][-1]

        # 更新持续时间
        self.gui_ecg_end_time.setText(str(self.ecg_end_time))
        self.gui_ecg_duration.setText(str(self.ecg_duration))

        # 切片数据
        self.ecg_data = self.ecg_data[start_index:end_index + 1]

        print('Final Start Index:', start_index)
        print('Final End Index:', end_index)
        print('Final Shape:', self.ecg_data.shape)
        print('Final Duration:', self.ecg_duration)

        # 创建新的时间序列
        time_serial = np.linspace(self.ecg_start_time, self.ecg_end_time, len(self.ecg_data))
        self.ecg_data[:, 0] = time_serial

        print('ECG Data:', self.ecg_data.shape)

        # 按mmwave的时间线性插值
        self.ecg_data = self.interpolate_ecg()

        # 绘制ECG图
        self.ecg_plot.plot(self.ecg_data[:,0], self.ecg_data[:,1])
        self.ecg_plot.setMouseEnabled(x=True, y=False)  # 仅启用x轴的鼠标缩放
        self.ecg_plot.enableAutoRange(axis=pg.ViewBox.YAxis) # 设置y轴的autoRange
    
    def plot_phase(self, index):
        x, y, z = index
        data = self.data_array[x, y, z, :]

        # 计算相位
        h = np.angle(data)
        # 相位解卷绕
        h = np.unwrap(h)
        h_ = process.compute_second_derivative(h, 1/200)
        self.h_ = h_

        # 绘制
        ## Phase
        self.phase_plot.clear()
        self.phase_plot.plot(self.mmwave_time, h)
        self.phase_plot.setTitle(f"Phase Plot at ({x}, {y}, {z})")
        self.phase_plot.setMouseEnabled(x=True, y=False)
        self.phase_plot.enableAutoRange(axis=pg.ViewBox.YAxis)
        ## Second Derivative
        self.second_derivative_plot.clear()
        self.second_derivative_plot.plot(self.mmwave_time, h_)
        self.second_derivative_plot.setTitle(f"Second Derivatived Plot at ({x}, {y}, {z})")
        self.second_derivative_plot.setMouseEnabled(x=True, y=False)  # 仅启用x轴的鼠标缩放
        self.second_derivative_plot.enableAutoRange(axis=pg.ViewBox.YAxis) # 设置y轴的autoRange
        ## 添加region,设置固定长度
        self.region = pg.LinearRegionItem([self.mmwave_time[0], self.mmwave_time[params['clip_time']*params['sampling_rate']-1]], brush=(0, 0, 255, 50))
        print('Region:', self.region.getRegion())
        self.region.setZValue(10)
        self.second_derivative_plot.addItem(self.region)
        self.region.sigRegionChanged.connect(self.update_region)

        # ECG图添加region
        if hasattr(self, 'ecg_region'):
            self.ecg_plot.removeItem(self.ecg_region)
        self.ecg_region = pg.LinearRegionItem([self.mmwave_time[0], self.mmwave_time[params['clip_time']*params['sampling_rate']-1]], brush=(0, 0, 255, 50))
        self.ecg_plot.addItem(self.ecg_region)
        self.ecg_region.sigRegionChanged.connect(self.index_match) # 微调

        # 初始化region_index
        self.region_index = [0, params['clip_time']*params['sampling_rate']]
        self.ecg_index = [0, params['clip_time']*params['sampling_rate']]

    def update_region(self, region):
        '''
        粗略匹配，将ECG的时间区间调整到与mmwave的时间区间一致
        '''
        region = self.region.getRegion()
        ecg_region = self.ecg_region.getRegion()
        # 计算当前长度
        current_length = region[1] - region[0]

        # 如果长度发生变化，则恢复到固定长度
        if current_length != params['clip_time']:
            # 恢复到固定长度
            center = (region[0] + region[1]) / 2
            self.region.setRegion([center - params['clip_time'] / 2, center + params['clip_time'] / 2])

        # 得到了region，应匹配ECG的时间
        print('Region:', region)
        print('start_time:', region[1] - region[0])
        self.ecg_region.setRegion([region[0], region[1]])

        # 保存现在的index至self中
        self.region_index = [int(region[0] * 200), int(region[1] * 200)]
        self.ecg_index = [int((region[0]+self.mmwave_start_time-self.ecg_start_time) * 200), int((region[1]+self.mmwave_start_time-self.ecg_start_time) * 200)]

    def index_match(self):
        # 拖拽ecg_region，实现微调
        ecg_region = self.ecg_region.getRegion()
        mmwave_region = self.region.getRegion()
        
        ecg_start = int((ecg_region[0]+self.mmwave_start_time-self.ecg_start_time) * 200)
        ecg_end = int((ecg_region[1]+self.mmwave_start_time-self.ecg_start_time) * 200)
        mmwave_start = int(mmwave_region[0] * 200)
        mmwave_end = int(mmwave_region[1] * 200)

        # 计算当前长度
        current_length = ecg_region[1] - ecg_region[0]

        # 如果长度发生变化，则恢复到固定长度
        if current_length != params['clip_time']:
            # 恢复到固定长度
            center = (ecg_region[0] + ecg_region[1]) / 2
            self.ecg_region.setRegion([center - params['clip_time'] / 2, center + params['clip_time'] / 2])

        # 得到了region，更新match_plot中的ecg数据
        ecg_data = self.ecg_data[ecg_start:ecg_end, 1]
        mmwave_data = self.h_[self.region_index[0]:self.region_index[1]]
        self.adjust_plot.clear()
        self.adjust_plot.plot(mmwave_data, pen='r')
        self.adjust_plot.plot(ecg_data, pen='g')

        # 保存目前的数据
        self.ecg_index = [ecg_start, ecg_end]
        self.region_index = [mmwave_start, mmwave_end]

    def output(self):
        # 导出目前index对应的数据
        mmwave_data = self.data_array[:, :, :, self.region_index[0]:self.region_index[1]]
        ecg_data = self.ecg_data[self.ecg_index[0]:self.ecg_index[1], 1]

        # 检查长度、是否有空值
        if mmwave_data.shape[3] != params['clip_time']*params['sampling_rate']:
            QMessageBox.warning(self, 'Warning', 'MMWave Length not match!')
            return
        if ecg_data.shape[0] != params['clip_time']*params['sampling_rate']:
            QMessageBox.warning(self, 'Warning', 'ECG Length not match!')
            return
        if np.isnan(mmwave_data).any():
            QMessageBox.warning(self, 'Warning', 'MMWave has NaN!')
            return
        if np.isnan(ecg_data).any():
            QMessageBox.warning(self, 'Warning', 'ECG has NaN!')
            return
        
        # 保存数据
        name = 'matched_' + self.gui_pkl_path_input.text().split('/')[-1]
        with open(name, 'wb') as f:
            pickle.dump([mmwave_data, ecg_data], f)
if __name__ == '__main__':
    # 全局参数
    global params
    params = {
        "sampling_rate": 200,
        'clip_time': 180
    }

    app = QApplication(sys.argv)
    ex = VisualizationApp()
    ex.show()
    sys.exit(app.exec_())
