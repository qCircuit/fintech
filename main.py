import configur
import ticker
import train

if __name__ == "__main__":


    repsol = ticker.Ticker("REP.MC")
    if configur.data_mode == "download":
        repsol.yf_download(
            configur.start_date, 
            configur.end_date, 
            configur.interval
        )
    elif configur.data_mode == "load":
        repsol.upload_data(
            configur.start_date, 
            configur.end_date
        )

    repsol.get_smooth()
    repsol.get_support()
    repsol.get_resistance()
    repsol.get_moving_average()
    repsol.get_stochastic_oscillator(configur.oscillator_period)

    # model, history = train.fit_convolution(repsol.data)
    model, history = train.fit_transformer(repsol.data) 
    print(history)
 
