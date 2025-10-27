#!/bin/bash

# Usage: ./delete_optuna_studies.sh <study_names>

STUDY_NAMES="$@"

# Database connection details
DB_USER="optuna"
DB_PASS="optuna_pw"
DB_NAME="optuna_db"
DB_HOST="localhost"

if [ -z "$STUDY_NAMES" ]; then
  echo "Usage: $0 <study_names>"
  exit 1
fi

for study_name in $STUDY_NAMES; do 

  STUDY_ID=$(mysql -u"$DB_USER" -p"$DB_PASS" -h "$DB_HOST" "$DB_NAME" -N -B <<SQL
  SELECT study_id FROM studies WHERE study_name = '$study_name';
SQL
  )

  echo "Deleting study_id = $STUDY_ID from database $DB_NAME..."

  mysql -u"$DB_USER" -p"$DB_PASS" -h "$DB_HOST" "$DB_NAME" <<SQL
  DELETE FROM trial_values WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trial_intermediate_values WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trial_system_attributes WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trial_user_attributes WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trial_heartbeats WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trial_params WHERE trial_id IN (
    SELECT trial_id FROM trials WHERE study_id = $STUDY_ID
  );
  DELETE FROM trials WHERE study_id = $STUDY_ID;

  DELETE FROM study_system_attributes WHERE study_id = $STUDY_ID;
  DELETE FROM study_directions WHERE study_id = $STUDY_ID;
  DELETE FROM study_user_attributes WHERE study_id = $STUDY_ID;

  DELETE FROM studies WHERE study_id = $STUDY_ID;
SQL

  echo "Study $STUDY_ID and associated data deleted successfully."

done

