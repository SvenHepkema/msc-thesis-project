SELECT unpacker, patcher, AVG(execution_time) as avg_execution_time FROM 'alp-data-ec.csv' WHERE n_vals == 1 and n_vecs == 1 GROUP BY unpacker, patcher ORDER BY avg_execution_time;
