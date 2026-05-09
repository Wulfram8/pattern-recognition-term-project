#!/bin/bash
set -e

echo "Running migrations..."
python manage.py makemigrations recognition
python manage.py migrate --noinput

echo "Seeding identities..."
python manage.py seed_identities

echo "Creating superuser..."
python manage.py createsuperuser --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput

echo "Starting server..."
exec gunicorn web.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120 \
    --access-logfile -
