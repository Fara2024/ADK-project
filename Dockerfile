# مرحله 1: ایمیج پایه
FROM python:3.11-slim

# مرحله 2: متغیرهای محیطی
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /usr/src/app

# مرحله 3: تنظیم دایرکتوری کاری
WORKDIR $APP_HOME

# مرحله 4: کپی کردن و نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# مرحله 5: کپی کردن تمام فایل‌های پروژه (شامل کدها و مدل‌ها)
COPY . $APP_HOME

# مرحله 6: افشای پورت
EXPOSE 8000

# مرحله 7: دستور پیش‌فرض اجرا
# در زمان اجرا توسط docker-compose، این CMD نادیده گرفته شده و با 'command' در فایل YML جایگزین می‌شود.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]