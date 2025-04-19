SELECT vbw, unpacker, unpack_n_vectors, duration_ns
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY vbw ORDER BY duration_ns ASC) AS rn
    FROM 'ffor.csv' WHERE data_type = 'u32' and unpacker != 'old_fls' and unpack_n_vectors = 4 and kernel = 'query'
) ranked
WHERE rn = 1 ORDER BY vbw ASC;
