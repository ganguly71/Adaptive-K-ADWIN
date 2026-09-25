#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define MAX_YEARS 1000
#define MAX_W_EFF 1000

#define DELTA 0.30
#define MIN_ADWIN_WINDOW 6
#define MIN_SUBWINDOW 3


/* ============================================================
                         INPUT DATA
   ============================================================ */

double data[] =
{
    57.89,
    57.33,
    58.40,
    23.00,
    56.00,
    78.00
};

#define N (sizeof(data) / sizeof(data[0]))


/* ============================================================
                         ADWIN STRUCTURE
   ============================================================ */

typedef struct
{
    double values[MAX_YEARS];
    size_t size;

} ADWIN;


/* ============================================================
                       KALMAN STRUCTURE
   ============================================================ */

typedef struct
{
    double x;
    double P;

} KalmanState;


/* ============================================================
                       ADWIN FUNCTIONS
   ============================================================ */

void adwin_init(ADWIN *window)
{
    window->size = 0;
}


int adwin_add(ADWIN *window, double value)
{
    if (window->size >= MAX_YEARS)
        return 0;

    window->values[window->size] = value;
    window->size++;

    return 1;
}


double range_mean(
    const ADWIN *window,
    size_t start,
    size_t end
)
{
    if (end <= start)
        return 0.0;

    double sum = 0.0;

    for (size_t i = start; i < end; i++)
    {
        sum += window->values[i];
    }

    return sum / (double)(end - start);
}


double range_variance(
    const ADWIN *window,
    size_t start,
    size_t end
)
{
    size_t n = end - start;

    if (n <= 1)
        return 0.0;

    double mean =
        range_mean(window, start, end);

    double sum = 0.0;

    for (size_t i = start; i < end; i++)
    {
        double difference =
            window->values[i] - mean;

        sum += difference * difference;
    }

    return sum / (double)(n - 1);
}


double adwin_epsilon(
    size_t n0,
    size_t n1,
    double delta
)
{
    double harmonic =
        (1.0 / (double)n0) +
        (1.0 / (double)n1);

    double log_term =
        log(4.0 / delta);

    return sqrt(
        0.5 *
        harmonic *
        log_term
    );
}


int adwin_detect_change(
    const ADWIN *window,
    size_t *cut_position
)
{
    size_t W = window->size;

    if (W < MIN_ADWIN_WINDOW)
        return 0;

    for (
        size_t cut = MIN_SUBWINDOW;
        cut <= W - MIN_SUBWINDOW;
        cut++
    )
    {
        size_t n0 = cut;
        size_t n1 = W - cut;

        double mean0 =
            range_mean(
                window,
                0,
                cut
            );

        double mean1 =
            range_mean(
                window,
                cut,
                W
            );

        double difference =
            fabs(mean0 - mean1);


        double variance =
            range_variance(
                window,
                0,
                W
            );

        double standard_deviation =
            sqrt(variance);


        if (standard_deviation < 0.01)
            standard_deviation = 0.01;


        double epsilon =
            adwin_epsilon(
                n0,
                n1,
                DELTA
            );


        double threshold =
            epsilon *
            standard_deviation;


        if (threshold < 0.5)
            threshold = 0.5;


        if (difference > threshold)
        {
            *cut_position = cut;

            return 1;
        }
    }

    return 0;
}


void adwin_shrink(
    ADWIN *window,
    size_t cut
)
{
    if (
        cut == 0 ||
        cut >= window->size
    )
    {
        return;
    }


    size_t new_size =
        window->size - cut;


    memmove(
        window->values,
        window->values + cut,
        new_size * sizeof(double)
    );


    window->size = new_size;
}


/* ============================================================
                       KALMAN FUNCTIONS
   ============================================================ */

void kalman_init(
    KalmanState *state,
    double initial_value
)
{
    state->x = initial_value;
    state->P = 1.0;
}


/*
    W_eff is the parameter being optimized.

    Q = 200 / W_eff

    R = W_eff^2 / 50
*/

double calculate_Q(double W_eff)
{
    return 200.0 / W_eff;
}


double calculate_R(double W_eff)
{
    return
        (W_eff * W_eff) /
        50.0;
}


double kalman_update(
    KalmanState *state,
    double measurement,
    double Q,
    double R
)
{
    /*
        Prediction step
    */

    double x_pred =
        state->x;

    double P_pred =
        state->P + Q;


    /*
        Kalman gain
    */

    double K =
        P_pred /
        (P_pred + R);


    /*
        Innovation
    */

    double innovation =
        measurement - x_pred;


    /*
        Update state
    */

    double x_updated =
        x_pred +
        K * innovation;


    double P_updated =
        (1.0 - K) *
        P_pred;


    if (P_updated < 0.0)
        P_updated = 0.0;


    state->x =
        x_updated;

    state->P =
        P_updated;


    return x_updated;
}


/* ============================================================
                 EVALUATE ONE W_eff VALUE
   ============================================================ */

double evaluate_window(
    double W_eff,
    double *total_loss,
    size_t *number_of_predictions
)
{
    /*
        Create fresh ADWIN for every W_eff.

        This is important because every candidate
        must start from exactly the same initial state.
    */

    ADWIN window;

    adwin_init(&window);


    /*
        Create fresh Kalman filter.
    */

    KalmanState kalman;

    kalman_init(
        &kalman,
        data[0]
    );


    /*
        A0 is the initial observation.
    */

    adwin_add(
        &window,
        data[0]
    );


    /*
        Calculate Q and R for this W_eff.
    */

    double Q =
        calculate_Q(W_eff);

    double R =
        calculate_R(W_eff);


    /*
        Reset loss.
    */

    *total_loss = 0.0;

    *number_of_predictions = 0;


    /*
        Start from A1.

        A1 is observed.
        Then the model is updated.
        Then P1 is produced.
        P1 is compared with A2.

        A2 is observed.
        Then P2 is produced.
        P2 is compared with A3.

        And so on.
    */

    for (
        size_t year = 1;
        year < N;
        year++
    )
    {
        /*
            Current observed value.
        */

        double current_value =
            data[year];


        /*
            Add current observation
            to ADWIN.
        */

        adwin_add(
            &window,
            current_value
        );


        /*
            Check ADWIN for change.
        */

        size_t cut_position = 0;

        int change_detected =
            adwin_detect_change(
                &window,
                &cut_position
            );


        /*
            If change is detected,
            remove the older portion.
        */

        if (change_detected)
        {
            adwin_shrink(
                &window,
                cut_position
            );
        }


        /*
            Update Kalman filter using
            the CURRENT observed value.
        */

        double prediction =
            kalman_update(
                &kalman,
                current_value,
                Q,
                R
            );


        /*
            Do not compare P with the
            current observation.

            Compare P with the NEXT
            year's actual value.

            P1 -> A2
            P2 -> A3
            P3 -> A4
            ...
        */

        if (year + 1 < N)
        {
            double actual_next =
                data[year + 1];


            /*
                Absolute prediction error.
            */

            double loss =
                fabs(
                    prediction -
                    actual_next
                );


            *total_loss += loss;

            (*number_of_predictions)++;
        }
    }


    /*
        Prevent division by zero.
    */

    if (*number_of_predictions == 0)
        return DBL_MAX;


    /*
        Mean Absolute Prediction Loss.
    */

    return
        *total_loss /
        (double)(*number_of_predictions);
}


/* ============================================================
                            MAIN
   ============================================================ */

int main(void)
{
    /*
        Validate number of observations.
    */

    if (N < 3)
    {
        printf(
            "At least 3 yearly averages are required.\n"
        );

        return EXIT_FAILURE;
    }


    if (N > MAX_YEARS)
    {
        printf(
            "Too many yearly averages.\n"
        );

        return EXIT_FAILURE;
    }


    /*
        Header
    */

    printf("\n");

    printf(
        "============================================================\n"
    );

    printf(
        "        K-ADWIN WINDOW SIZE OPTIMIZATION\n"
    );

    printf(
        "============================================================\n"
    );


    /*
        Input data
    */

    printf("\nInput Data:\n");

    printf(
        "------------------------------------------------------------\n"
    );


    for (size_t i = 0; i < N; i++)
    {
        printf(
            "A%zu = %.4f\n",
            i,
            data[i]
        );
    }


    /*
        Explain prediction sequence.
    */

    printf("\nPrediction structure:\n");

    printf(
        "A1 -> P1 -> compare P1 with A2\n"
    );

    printf(
        "A2 -> P2 -> compare P2 with A3\n"
    );

    printf(
        "A3 -> P3 -> compare P3 with A4\n"
    );

    printf(
        "...\n"
    );


    /*
        Results table.
    */

    printf("\n");

    printf(
        "============================================================\n"
    );

    printf(
        "                  WINDOW SIZE RESULTS\n"
    );

    printf(
        "============================================================\n"
    );


    printf(
        "%-12s %-18s %-18s\n",
        "W_eff",
        "Total Loss",
        "Average Loss"
    );


    printf(
        "------------------------------------------------------------\n"
    );


    /*
        Best result found so far.
    */

    double best_W =
        1.0;

    double best_average_loss =
        DBL_MAX;

    double best_total_loss =
        DBL_MAX;


    /*
        Number of consecutive increases.

        We stop after two consecutive
        increases in average loss.
    */

    int increasing_count =
        0;


    /*
        Search W_eff.

        W_eff starts from 1 and increases
        until the loss begins increasing
        consecutively.
    */

    for (
        int w = 1;
        w <= MAX_W_EFF;
        w++
    )
    {
        double W_eff =
            (double)w;


        double total_loss;

        size_t number_of_predictions;


        /*
            Evaluate this W_eff.
        */

        double average_loss =
            evaluate_window(
                W_eff,
                &total_loss,
                &number_of_predictions
            );


        /*
            Print result.
        */

        printf(
            "%-12.0f %-18.6f %-18.6f\n",
            W_eff,
            total_loss,
            average_loss
        );


        /*
            Check if this is the
            new minimum loss.
        */

        if (
            average_loss <
            best_average_loss
        )
        {
            /*
                New best W_eff.
            */

            best_average_loss =
                average_loss;

            best_total_loss =
                total_loss;

            best_W =
                W_eff;


            /*
                Loss improved, so reset
                consecutive-increase count.
            */

            increasing_count = 0;
        }
        else
        {
            /*
                Loss did not improve.

                Since W_eff is increasing,
                treat this as an increase
                from the best/minimum region.
            */

            increasing_count++;


            /*
                Stop after two consecutive
                non-improving values.
            */

            if (
                increasing_count >= 2
            )
            {
                printf("\n");

                printf(
                    "Average loss has started increasing.\n"
                );

                printf(
                    "Search stopped after two "
                    "consecutive non-improvements.\n"
                );

                break;
            }
        }
    }


    /*
        Calculate Q and R for the
        selected W_eff.
    */

    double best_Q =
        calculate_Q(best_W);

    double best_R =
        calculate_R(best_W);


    /*
        Print final result.
    */

    printf("\n");

    printf(
        "============================================================\n"
    );

    printf(
        "                    BEST WINDOW\n"
    );

    printf(
        "============================================================\n"
    );


    printf(
        "Best W_eff              : %.0f\n",
        best_W
    );


    printf(
        "Total Prediction Loss   : %.6f\n",
        best_total_loss
    );


    printf(
        "Average Prediction Loss : %.6f\n",
        best_average_loss
    );


    printf(
        "Q                       : %.6f\n",
        best_Q
    );


    printf(
        "R                       : %.6f\n",
        best_R
    );


    printf(
        "============================================================\n"
    );


    printf("\n");


    printf(
        "The selected W_eff is the value producing\n"
    );

    printf(
        "the minimum average one-year-ahead prediction loss\n"
    );

    printf(
        "before the loss begins to increase.\n"
    );


    return EXIT_SUCCESS;
}