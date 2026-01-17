\# Ping Sweep Detection using Machine Learning



โปรเจกต์นี้มีจุดประสงค์เพื่อพัฒนาระบบตรวจจับการสแกนเครือข่าย (Ping Sweep) โดยประยุกต์ใช้เทคนิค Machine Learning โดยเปรียบเทียบประสิทธิภาพระหว่าง \*\*Random Forest\*\* และ \*\*XGBoost\*\* โดยระบบจะทำการเลือกโมเดลที่ให้ผลลัพธ์แม่นยำที่สุด (Best Model) มาใช้ในการทำนายและวิเคราะห์ข้อมูลโดยอัตโนมัติ



\## ลักษณะชุดข้อมูล (Dataset)



\*\*ไฟล์:\*\* `ping\_sweep\_enterprise.csv`



\### รายละเอียด Feature (คอลัมน์)

ข้อมูลประกอบด้วย 8 คอลัมน์หลัก:

1\.  \*\*timestamp:\*\* เวลาที่เกิดเหตุการณ์ 

2\.  \*\*src\_ip:\*\* IP ต้นทาง 

3\.  \*\*dst\_ip:\*\* IP ปลายทาง 

4\.  \*\*protocol:\*\* โปรโตคอลที่ใช้ (ICMP, TCP, UDP)

5\.  \*\*flag:\*\* สถานะ Flag ของ Packet (เช่น SYN, ACK, PSH)

6\.  \*\*packet\_size:\*\* ขนาดของ Packet (Bytes) 

7\.  \*\*ttl:\*\* ค่า Time-To-Live

8\.  \*\*label:\*\* ผลเฉลย (Target)

&nbsp;   \* `0`: Normal (ปกติ)

&nbsp;   \* `1`: Attack (Ping Sweep)



\##  การทำงานของโค้ด (Workflow)



ไฟล์ `Ai.py` มีขั้นตอนการทำงานดังนี้:



\### 1. Data Loading \& Cleaning

\* ลบช่องว่างหน้า-หลังในชื่อคอลัมน์ (`.strip()`) เพื่อป้องกัน Error



\### 2. Preprocessing

\* คัดเลือก Feature ที่จำเป็น และตัดข้อมูลที่ไม่ใช่นัยสำคัญออก

\* แปลงข้อมูลตัวอักษร (Categorical) เป็นตัวเลขด้วย `LabelEncoder`

\* แบ่งข้อมูลเป็นชุด Train (70%) และ Test (30%)



\### 3. Model Training (Dual Models)

ระบบจะเทรน 2 โมเดลพร้อมกัน:

\* \*\*Random Forest Classifier:\*\* โมเดลตระกูล Decision Tree

\* \*\*XGBoost Classifier:\*\* โมเดลตระกูล Gradient Boosting



\### 4. Auto-Selection \& Evaluation

\* เปรียบเทียบค่า \*\*Accuracy\*\* ของทั้งสองโมเดล

\* เลือกโมเดลที่ชนะ (Winner) มาแสดงผลลัพธ์:

&nbsp;   \* \*\*Confusion Matrix:\*\* ตารางความแม่นยำ

&nbsp;   \* \*\*Classification Report:\*\* ค่า Precision, Recall, F1-Score

&nbsp;   \* \*\*Feature Importance:\*\* กราฟวงกลมแสดงน้ำหนักปัจจัยที่ใช้ตัดสินใจ



\### 5. Visualization

แสดงกราฟวิเคราะห์ข้อมูลดิบเชิงลึก:

\* \*\*Protocol \& Flag Analysis:\*\* เปรียบเทียบความถี่การใช้งาน

\* \*\*Packet Size \& TTL Analysis:\*\* แผนภาพ Boxplot และ Violin Plot เพื่อดูการกระจายตัวของข้อมูล





