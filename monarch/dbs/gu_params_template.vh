parameter INPUT_NUM = <input_num>;
parameter OUTPUT_NUM = <output_num>;

parameter PIPE_DEPTH = <pipe_depth>;

parameter SOURCE_NUM = <source_num>;
parameter SINK_NUM = <sink_num>;

parameter integer ARR_PREDELAY [0:SOURCE_NUM-1] = '{
    <arr_predelay>
};

// Integer codings for the type of input to the source node.
parameter integer ARR_SOURCE_TYPE [0:SOURCE_NUM-1] = '{
    <arr_source_type>
};

// Index of 1st sink node to be routed to a given source node.
parameter integer ARR_IN_1 [0:SINK_NUM-1] = '{
    <arr_in_1>
};

// Input delay value after predelay into 1st input of given sink node
parameter integer ARR_DELAY_1 [0:SINK_NUM-1] = '{
    <arr_dly_1>
};

// Index of 2nd sink node to be routed to a given source node.
parameter integer ARR_IN_2 [0:SINK_NUM-1] = '{
    <arr_in_2>
};

// Input delay value after predelay into 2nd input of given sink node
parameter integer ARR_DELAY_2 [0:SINK_NUM-1] = '{
    <arr_dly_2>
};

/*
// Source node index for matrix route.
parameter integer ARR_ROUTE_SOURCE [SOURCE_NUM-1:0] = '{
    <arr_route_source>
};

// Sink node index for matrix route.
parameter integer ARR_ROUTE_SINK [SINK_NUM-1:0] = '{
    <arr_route_sink>
};
*/