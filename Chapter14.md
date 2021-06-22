# Chapter 14 CNN
## The Architecture of the Visual Cortex
Inspired by the study of structures of the visual cortex

## Convolutional Layers
1. Convolutional Layers connect to pixels in their receptive fields
2. In a CNN, the input images can be 2D
3. CNN, receptive fields: width, height, strides
4. Filters(convolution kernels)
5. Stacking Multiple Feature Maps: Figure 14_6画的不错 <font color="#dd0000">这边需要再对照中文版理解下</font>
6. Tensorflow implementation：keras.layers.Conv2D(filters=, kernel_size=, strids=, padding='', activation='')
8. Memory requirements: <font color="#dd0000">需要再理解下如何计算内存消耗的</font>

## Pooling Layers
1. Performs just like convolutional layers, but without weights, only apllied by an aggregation function such as max or mean.
2. max pooling layers: can provide some translation invariance, even a small amount of rotational invariance and a slight scale invariance.
3. Tensorflow implementation: max_pool/ave_pool = keras.layers.MaxPool2D/AvgPool2D(pool_size=2), strides(default)=kernel size. Generally, max pool performs better than Avg pool, since it enhance the strongest signals. Depthwise max pool: tf.nn.max_pool()

## CNN Architectures
1. Few convonlution layers (ReLUs) --> Pooling layer --> Few convolution layers (ReLUs) --> Pooling layer --> Fully connected (ReLUs)
2. 

class pix_label(QLabel):
    def __init__(self,parent):
        super(pix_label,self).__init__(parent)
        self.setMouseTracking(True)
        self.pix = QPixmap(768, 768)#考虑边框的间距 减去px
        self.pix.fill(Qt.black)
        self.setStyleSheet("border: 2px solid red")
        self.erase_img = False
        self.lastPoint = QPoint()
        self.endPoint  = QPoint()
        self.drawPoint = QPoint()
        
    def paintEvent(self, event): # 重写paintEvent事件
        pen = QPen(Qt.blue, 1.5, Qt.DashLine)
        if not self.erase_img:
            Line_draw = QPainter(self.pix)
            Line_draw.setPen(pen)
            Line_draw.drawLine(self.lastPoint, self.endPoint)
            painter = QPainter(self)
            painter.drawPixmap(2,2,self.pix)
        else:
            erase = QPen(Qt.white, 1.5, Qt.DashLine)
            Line_draw = QPainter(self.pix)
            Line_draw.setPen(erase)
            Line_draw.drawLine(self.lastPoint, self.endPoint)
            painter = QPainter(self)
            painter.drawPixmap(2,2,self.pix)
            self.erase_img = False
        if self.drawPoint:
            Line_draw = QPainter(self)
            Line_draw.setPen(pen)
            Line_draw.drawLine(self.drawPoint.x()+2, self.drawPoint.y()+2, self.lastPoint.x()+2, self.lastPoint.y()+2)    
        
        # self.MainUI.pixmap.setPixmap(self.pix)

    def erase(self):
        self.erase_img = True
        self.update()
        
    def mousePressEvent(self, event): # 重写鼠标按下事件
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint
 
    def mouseMoveEvent(self, event):  # 重写鼠标移动事件
        if event.buttons() and Qt.LeftButton:
            self.drawPoint = event.pos()
            self.update()             # 更新绘图事件,每次执行update都会触发一次paintEvent(self, event)函数
 
    def mouseReleaseEvent(self, event): #重写鼠标释放事件
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            self.update()
        
class MainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setMouseTracking(True)
        self.beginpoint = QPoint()
        self.endpoint   = QPoint()
        self.bot        = 0
        self.label      = pix_label(self)
        self.label.setGeometry(QtCore.QRect(240, 200, 772, 772))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.imgfile    = ''
        # self.first_rotated = 1
        # self.first_rotated = 1
        
        ### 添加pyqtgraph组件
        self.h_line = pg.PlotWidget(self)
        self.h_line.setGeometry(QtCore.QRect(220, 12, 792, 150))
        self.v_line = pg.PlotWidget(self)
        self.v_line.setGeometry(QtCore.QRect(62, 200, 150, 792))
        self.v_curve = self.v_line.plot(np.zeros(2048),np.arange(2048), name='model1')
        self.h_curve = self.h_line.plot(np.arange(2048),np.zeros(2048), name='model1')
       
        
        ### 链接信号
        self.IF.clicked.connect(self.openfile)
        self.AH.clicked.connect(self.auto_horizon)
        self.Top.clicked.connect(self.define_top)
        self.Bot.clicked.connect(self.define_bot)
        self.Top_pos_adjust.valueChanged.connect(self.define_top)
        self.Bot_pos_adjust.valueChanged.connect(self.define_bot)
        self.MCD.clicked.connect(self.measure_cd)
        
        # 滑块链接画灰度曲线
        self.horizontalSlider.valueChanged.connect(self.draw_v_curve)
        self.verticalSlider.valueChanged.connect(self.draw_h_curve)

        
    # 加载图片
    def openfile(self):
        fname=QFileDialog.getOpenFileName(self,"选择图片文件",".")
        self.imgfile = fname[0]
        if fname[0]:
            img = cv2.imread(fname[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.img = img
            self.origin_img = img.copy()
            self.img_update() 
            self.Output.setText('加载图片完毕')
        self.first_bot     = 1
        self.first_top     = 1
        self.first_h_index = 1
        self.first_v_index = 1
        self.first_rotated = 1
        return self.img
   
    # 每次更新图片
    def img_update(self):
        img = Image.fromarray(self.img)
        outshape = (768, 768)
        img = img.resize(outshape, Image.NEAREST)
        img_pix = img.toqpixmap()
        self.label.pix=img_pix
        self.label.update()
    
    # 根据label内的press和release坐标的点，确定直线
    def horizon_line(self):
        p0 = self.label.lastPoint
        p1 = self.label.endPoint
        k = (p1.y() - p0.y())/(p1.x() - p0.x())
        b = p1.y() - k*p1.x()
        return (k, b)
    
    # 旋转图片   
    def auto_horizon(self):
        a0 = PrintTime('修正图片方向'); a0.start()
        k, b = self.horizon_line()
        img = self.img
        a1 = PrintTime('旋转'); a1.start()
        M = cv2.getRotationMatrix2D((0,0), -math.degrees(-np.arctan(k)),1)
        img_rotated = cv2.warpAffine(img, M, img.shape)
        a1.end()
        
        a2 = PrintTime('平移'); a2.start()
        v_shift = np.array([[1,0,0],[0,1,(780-b)*8/3]])
        img_rotated_vshift = cv2.warpAffine(img_rotated, v_shift, img_rotated.shape)
        img_rotated_vshift1 = cv2.warpAffine(img_rotated, v_shift, img_rotated.shape)
        a2.end()
        
        # np.save(os.path.splitext(self.imgfile)[0] + '_rotated', img_rotated_vshift, allow_pickle=True)
       
        self.img = img_rotated_vshift
        self.normal_img = img_rotated_vshift1.copy()
        # self.first_rotated = 1
        # self.label.erase()
        self.img_update()
        self.Output.append('Auto horizon 完毕')
        self.first_rotated = 0
        a0.end()
    
    def draw_v_curve(self):
        self.draw_curve(0)
    
    def draw_h_curve(self):
        self.draw_curve(1)
        
    def draw_curve(self, linetype=0):
        if self.first_rotated:
            img = self.origin_img
        else:
            img = self.normal_img
        self.hw   = int(int(self.width.value())//2)
        v_pos = self.horizontalSlider.value()
        v_line= np.mean(img[:, max(v_pos-self.hw, 0) : min(v_pos+self.hw+1, 2048)], axis=1)
        h_pos = self.verticalSlider.value()
        h_line= np.mean(img[max(2047-h_pos-self.hw, 0) : min(2048-h_pos+self.hw, 2048), :], axis=0)        
        if linetype==0:
            self.v_curve.setData(v_line[::-1], np.arange(2048))
            self.draw_line(img, v_pos, 1, self.first_v_index)            
            # self.draw_line(img, h_pos, 0, 1)
            self.first_v_index = 0
        else:
            self.h_curve.setData(np.arange(2048), h_line)
            self.draw_line(img, h_pos, 0, self.first_h_index)
            # self.draw_line(img, v_pos, 1, 1)  
            self.first_h_index = 0
        
    # 定义critical位置
    def define_bot(self):
        self.bot = int(self.Bot_pos.text()) + int(self.Bot_pos_adjust.value())
        pre_img  = self.normal_img.copy()
        self.draw_line(pre_img, self.bot, 0, self.first_bot)
        # self.first_bot = 0
        
    def define_top(self):
        self.top = int(self.Top_pos.text()) + int(self.Top_pos_adjust.value())
        pre_img  = self.img.copy()
        self.draw_line(pre_img, self.top, 0, self.first_top)
        self.first_top = 0
        
    # 在PixMap上画黑线标示当前定义位置
    def draw_line(self, pre_img, pos, linetype=0, first_draw=1):
        if first_draw:
            img = self.img
            self.Output.append('1')
        else:
            img = pre_img
            self.Output.append('0')
        row, col = img.shape
        if linetype==0:
            cv2.line(img, (0, row - pos), (col, row - pos), (0, 255, 0), thickness=1)
        else:
            cv2.line(img, (pos, 0), (pos, col), (0, 255, 0), thickness=1)
        self.img = img
        self.img_update()
        self.Output.append('设置Bottom位置为{}'.format(self.bot))
        self.first_bot = 0
    
    # 量测CD的方法函数
    def measure_cd(self):
        img = self.img[::-1,:]
        row, col = img.shape  
        itemlist = ['CD_Top','CD_Mid', 'CD_Bot']
        # 填充0区域，都等于最近非0值
        for j in range(row):
            line = img[j,:]
            idx = np.arange(col)[line==0]
            if len(idx)!=0:
                line[idx] = line[idx[0]-15 : idx[0]].mean()
       
        # # 只取三行
        for i, item in enumerate(itemlist):
            try:
                h = int(self.CD_tab.item(i, 0).text()) + self.bot
                w = int(self.CD_tab.item(i, 1).text())
                     
                # 根据定义宽度求灰度曲线的平均值并做平滑处理
                grey_line = img[int(h - w/2):int(h + w/2), :]
                grey_line = np.sum(grey_line, axis=0)
                # grey_line[(idx[0]-20):(idx[0]-1)].mean()
                grey_line = signal.savgol_filter(grey_line, 31, 3)
         
                # 为了求取平滑的梯度，避免局部variation大使得梯度异常，选择在左右延展10个pixel的空间上计算当前位置的梯度
                deltarange = 10
                padding_line = np.ones(col + 2*deltarange)
                padding_line[deltarange : col + deltarange] = grey_line
                padding_line[:deltarange] = padding_line[deltarange]
                padding_line[col + deltarange:] = padding_line[col + deltarange - 1]
                
                # 求梯度
                grads = []
                for j in range(col):
                    grad = (padding_line[j+deltarange+1:j+deltarange+11][::-1] - padding_line[j+deltarange-10:j+deltarange])*np.array([1/(2*(k+1)) for k in range(deltarange)]).mean()
                    grad = grad.mean()
                    grads.append(grad)
                    
                grads = np.array(grads)
                grads = np.abs(grads)
                
                # 根据梯度大小判断边界
                big_grad    = np.arange(col)[grads>1.68*w]
                bound_right = np.hstack([big_grad, [col+1]])[1:] - np.hstack([big_grad, [col+1]])[:-1]
                bound_left  = np.hstack([[-2],big_grad])[1:] - np.hstack([[-2], big_grad])[:-1]
                right_idx   = np.arange(len(bound_right))[bound_right!=1]
                left_idx    = np.arange(len(bound_left))[bound_left!=1]
                bound_idx   = ((big_grad[right_idx] + big_grad[left_idx])/2).astype(int)
                cd          = bound_idx[1:] - bound_idx[:-1]
                cd_pos      = (bound_idx[1:] + bound_idx[:-1])/2
                h_img       = 2048 - h
                
                for j, pos in enumerate(cd_pos):
                    pos = int(pos)
                    self.img = cv2.arrowedLine(self.img, (pos+20,h_img), (pos + int( cd[j]/2-20), h_img), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.1)
                    self.img = cv2.arrowedLine(self.img, (pos-20,h_img), (pos + int(-cd[j]/2+20), h_img), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.1)
                    self.img = cv2.putText(self.img, str(cd[j]), (pos-40, h_img-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 10)
                
                self.Output.append(item + '量测完毕')
            except:
                self.Output.append(item + '未输入位置')
        self.img_update()

    # 量测height的方法函数
    def measure_h(self):
        img = self.img[::-1,:]
        row, col = img.shape  
        itemlist = ['CD_Top','CD_Mid', 'CD_Bot']
        # 填充0区域，都等于最近非0值
        for j in range(row):
            line = img[j,:]
            idx = np.arange(col)[line==0]
            if len(idx)!=0:
                line[idx] = line[idx[0]-15 : idx[0]].mean()
       
        # # 只取三行
        for i, item in enumerate(itemlist):
            try:
                h = int(self.CD_tab.item(i, 0).text()) + self.bot
                w = int(self.CD_tab.item(i, 1).text())
                     
                # 根据定义宽度求灰度曲线的平均值并做平滑处理
                grey_line = img[int(h - w/2):int(h + w/2), :]
                grey_line = np.sum(grey_line, axis=0)
                # grey_line[(idx[0]-20):(idx[0]-1)].mean()
                grey_line = signal.savgol_filter(grey_line, 31, 3)
         
                # 为了求取平滑的梯度，避免局部variation大使得梯度异常，选择在左右延展10个pixel的空间上计算当前位置的梯度
                deltarange = 10
                padding_line = np.ones(col + 2*deltarange)
                padding_line[deltarange : col + deltarange] = grey_line
                padding_line[:deltarange] = padding_line[deltarange]
                padding_line[col + deltarange:] = padding_line[col + deltarange - 1]
                
                # 求梯度
                grads = []
                for j in range(col):
                    grad = (padding_line[j+deltarange+1:j+deltarange+11][::-1] - padding_line[j+deltarange-10:j+deltarange])*np.array([1/(2*(k+1)) for k in range(deltarange)]).mean()
                    grad = grad.mean()
                    grads.append(grad)
                    
                grads = np.array(grads)
                grads = np.abs(grads)
                
                # 根据梯度大小判断边界
                big_grad    = np.arange(col)[grads>1.68*w]
                bound_right = np.hstack([big_grad, [col+1]])[1:] - np.hstack([big_grad, [col+1]])[:-1]
                bound_left  = np.hstack([[-2],big_grad])[1:] - np.hstack([[-2], big_grad])[:-1]
                right_idx   = np.arange(len(bound_right))[bound_right!=1]
                left_idx    = np.arange(len(bound_left))[bound_left!=1]
                bound_idx   = ((big_grad[right_idx] + big_grad[left_idx])/2).astype(int)
                cd          = bound_idx[1:] - bound_idx[:-1]
                cd_pos      = (bound_idx[1:] + bound_idx[:-1])/2
                h_img       = 2048 - h
                
                for j, pos in enumerate(cd_pos):
                    pos = int(pos)
                    self.img = cv2.arrowedLine(self.img, (pos+20,h_img), (pos + int( cd[j]/2-20), h_img), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.1)
                    self.img = cv2.arrowedLine(self.img, (pos-20,h_img), (pos + int(-cd[j]/2+20), h_img), (0, 255, 0), thickness=3, line_type=cv2.LINE_4, shift=0, tipLength=0.1)
                    self.img = cv2.putText(self.img, str(cd[j]), (pos-40, h_img-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 10)
                
                self.Output.append(item + '量测完毕')
            except:
                self.Output.append(item + '未输入位置')
        self.img_update()
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
