import sys
import signal
import json

import dash_bootstrap_components as dbc
import flask
import pandas as pd
from dash import html, State, dcc, dash, callback_context
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform
from dash.exceptions import PreventUpdate

import kinect_control_sdk
import kinect_control_wrapper
import storage_control
import video_processing

# Dash Startups
server_flask = flask.Flask(__name__)
app = DashProxy(server=server_flask, transforms=[MultiplexerTransform()],
                external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create different Class-objects
kcSdk = kinect_control_sdk.KinectControlSDK()
kcWrappper = kinect_control_wrapper.KinectControlWrapper()
sc = storage_control.StorageControl()
vp = video_processing.VideoProcessing()

# Create Layout
app.layout = dbc.Container([
    dcc.Interval(id='interval1', interval=10 * 1000, n_intervals=0),
    dcc.Interval(id="interval_saveConfigSpan", interval=5000, max_intervals=1),
    dcc.Interval(id="interval_initial", n_intervals=0, max_intervals=0, interval=1),
    # Stores current config per session
    dcc.Store(id='store_config'),
    dbc.Row([
        html.H3("eGate Recording", className='text-center text-primary')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Row([
                    html.H3("Azure SDK"),
                    html.Div([
                        "Select FPS",
                        dcc.Dropdown(["30", "15", "5"], value="30", id="dropdown_fps", style={"marginBottom": "1.5em"})
                    ]),
                    html.Div([
                        "Select Depth Mode",
                        dcc.Dropdown(["NFOV_2X2BINNED", "NFOV_UNBINNED", "WFOV_2X2BINNED", "WFOV_UNBINNED",
                                      "PASSIVE_IR", "OFF"], value="NFOV_UNBINNED",
                                     id="dropdown_depth", style={"marginBottom": "1.5em"}),
                    ]),
                    html.Div([
                        "Select RGB-Resolution",
                        dcc.Dropdown(["3072p", "2160p", "1536p", "1440p", "1080p", "720p", "OFF"], value="720p",
                                     id="dropdown_color", style={"marginBottom": "1.5em"}),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Save Config", id="button_saveConfig", size="sm",
                                       style={'width': '160px', 'margin-right': '1.5em', "marginBottom": "1.5em"}),
                        ]),
                        dbc.Col([
                            html.Span("", id="span_saveConfig")
                        ])
                    ]),
                    html.Div([
                        "Current Probe-ID: ",
                        dcc.Input(value="ABC123", id="input_probeId", style={"marginBottom": "1.5em"})
                    ]),
                    dbc.Row([
                        html.Div([
                            dbc.Button("Start", id="button_start_sdk", size="sm",
                                       style={'width': '80px', 'margin-right': '1.5em', "marginBottom": "1.5em"}),
                            dbc.Button("Stop", id="button_stop_sdk", size="sm",
                                       style={"width": "80px", 'margin-right': '1.5em', "marginBottom": "1.5em"}),
                            dbc.Button("Exit Server", id="button_exit_server_sdk", size="sm",
                                       style={"width": "160px", 'margin-right': '1.5em', "marginBottom": "1.5em"})
                        ]),
                    ]),
                ]),
                dbc.Row([
                    html.Span(id="span_start_sdk"),
                    html.Span("", id="span_stop_sdk"),
                    html.Span("", id="span_exit_server_sdk")
                ]),
                dbc.Row([
                    html.Div([
                        dbc.Button("Live-Viewer", id="button_showVideo_wrapper", size="sm",
                                   style={"width": "360px", 'margin-right': '1.5em', "marginBottom": "1.5em"})
                    ])
                ])
                # dbc.Row([
                #     html.H3("Python Wrapper"),
                #     dbc.Button("Start", id="button_start_wrapper"),
                #     dbc.Button("Stop", id="button_stop_wrapper"),
                #     dbc.Button("Show Video", id="button_showVideo_wrapper")
                # ]),
                # dbc.Row([
                #     html.Span("", id="span_start_wrapper"),
                #     html.Span("", id="span_stop_wrapper"),
                # ]),
            ]),
            html.Span("", id="dummy")
        ]),
        dbc.Col([
            html.Div([
                html.H3("Storage Info"),
                dbc.Progress(id="progress_storage"),
                html.Span("", id="span_storage")
            ]),
            html.Div([
                html.H3("Video Postprocessing"),
                html.Span("Extract all video tracks from stored .mkv Files"),
                dbc.Button("Extract Video Tracks", id="button_extract_video_tracks"),
                html.Span("", id="span_extract_video_tracks")
            ])
        ])
    ])
])


# @app.callback(
#     Output(component_id="span_start_sdk", component_property="children"),
#     Input(component_id="button_start_sdk", component_property="n_clicks"),
#     State(component_id="input_probeId", component_property="value"),
#     State(component_id="dropdown_fps", component_property="value"),
#     State(component_id="dropdown_depth", component_property="value"),
#     State(component_id="dropdown_color", component_property="value")
# )
# def button_start_sdk_clicked(n_clicks, probeId, fps, depth, color):
#     kcSdk.record_sequences(probeId, fps, depth, color)
#     return "Sequence running"


# @app.callback(
#     Output(component_id="span_stop_sdk", component_property="children"),
#     Input(component_id="button_stop_sdk", component_property="n_clicks")
# )
# def button_stop_sdk_clicked(n_clicks):
#     sc.getStorageData()
#     kcSdk.stop_record()
#     return "Sequence stopped"

@app.callback(
    [Output(component_id="dummy", component_property="children"),
     Output(component_id="store_config", component_property="data")],
    Input(component_id="interval_initial", component_property="n_intervals"),
    State(component_id="dropdown_fps", component_property="value"),
    State(component_id="dropdown_depth", component_property="value"),
    State(component_id="dropdown_color", component_property="value")
)
def saveDefaultConfig(n_intervals, fps, depth, color):
    config = {
        'fps': fps,
        'depth': depth,
        'color': color
    }
    jsonConf = json.dumps(config, indent=4)
    print(jsonConf)
    return "", jsonConf


@app.callback(
    Output(component_id="span_exit_server_sdk", component_property="children"),
    Input(component_id="button_exit_server_sdk", component_property="n_clicks"),
    prevent_initial_call=True
)
def button_exit_server_sdk_clicked(n_clicks):
    sys.exit()


# @app.callback(
#     Output(component_id="span_start_wrapper", component_property="children"),
#     Input(component_id="button_start_wrapper", component_property="n_clicks")
# )
# def button_start_wrapper_clicked(n_clicks):
#     kcWrappper.record_sequence()
#     return "Sequence running"


# @app.callback(
#     Output(component_id="span_stop_wrapper", component_property="children"),
#     Input(component_id="button_stop_wrapper", component_property="n_clicks")
# )
# def button_stop_wrapper_clicked(n_clicks):
#     kcSdk.stop_record()
#     return "Sequence stopped"


@app.callback(
    Output(component_id="dummy", component_property="children"),
    Input(component_id="button_showVideo_wrapper", component_property="n_clicks"),
    prevent_initial_call=True
)
def button_showVideo_wrapper_clicked(n_clicks):
    kcWrappper.display_video()
    return ""


@app.callback([
    Output(component_id="span_storage", component_property="children"),
    Output(component_id="progress_storage", component_property="value"),
    Output(component_id="progress_storage", component_property="label")],
    Input(component_id="interval1", component_property="n_intervals"),
    prevent_initial_call=True
)
def button_update_storage(n_clicks):
    total, used, free = sc.getStorageData()
    span_text = f"Total: {str(round(total, 1))}GB, Used: {str(round(used, 1))}GB, Free: {str(round(free, 1))}GB"
    percantage_used = (used / total) * 100
    percantage_used = round(percantage_used, 1)
    percantage_used_string = str(percantage_used) + " %"
    return span_text, percantage_used, percantage_used_string


@app.callback(
    Output(component_id="span_extract_video_tracks", component_property="children"),
    Input(component_id="button_extract_video_tracks", component_property="n_clicks"),
    prevent_initial_call=True
)
def button_extract_video_tracks(n_clicks):
    # vp.convert_to_mp4()
    vp.startVideoPostprocess()
    return "Videos processed"


# Callback Funktion inkludiert Start & Stop, da so gemeinsam auf ein Ausgabefeld zugegriffen werden kann
@app.callback([Output('span_start_sdk', 'children')],
              [Input('button_start_sdk', 'n_clicks'),
               Input('button_stop_sdk', 'n_clicks'),
               Input('input_probeId', 'value'),
               Input('store_config', 'data')],
              prevent_initial_call=True
              )
def callbackStartStop(one, n_clicks, probeId, data):
    ctx = callback_context
    source = ctx.triggered[0]['prop_id'].split('.')[0]
    if source == 'button_start_sdk':
        resp = json.loads(data)
        fps = resp['fps']
        depth = resp['depth']
        color = resp['color']
        kcSdk.record_sequences(probeId, fps, depth, color)
        return 'Started'
    elif source == 'button_stop_sdk':
        kcSdk.stop_record()
        return 'Stopped'


# HTTP-Request Start
@server_flask.route("/start")
# @app.callback(Input(component_id="dropdown_fps", component_property="value"))
def startRecordHTTP():
    defaultProbeID = "123456789"
    #kcSdk.record_sequences(defaultProbeID, "30", "NFOV_UNBINNED", "720p")
    kcSdk.record_sequences(defaultProbeID, fps="15", depth="WFOV_UNBINNED", color="2160p")
    return "Started"


# HTTP-Request Stop
@server_flask.route("/stop/filename=<filename>")
# @app.callback(Input(component_id="dropdown_fps", component_property="value"))
def stopRecordHTTP(filename=None):
    kcSdk.stop_record(filename)
    return "Stopped"


# Save Config
@app.callback([Output(component_id="span_saveConfig", component_property="children"),
               Output(component_id="span_saveConfig", component_property="style"),
               Output(component_id="store_config", component_property="data")],
              Input(component_id="button_saveConfig", component_property="n_clicks"),
              State(component_id="dropdown_fps", component_property="value"),
              State(component_id="dropdown_depth", component_property="value"),
              State(component_id="dropdown_color", component_property="value"),
              prevent_initial_call=True
              )
def button_saveConfig(n_clicks, fps, depth, color):
    config = {
        'fps': fps,
        'depth': depth,
        'color': color
    }
    jsonConf = json.dumps(config, indent=4)
    print(jsonConf)
    return "Saved!", {"display": "block"}, jsonConf


# Let "Save!" Text disappear after 5 sec
@app.callback(Output(component_id="span_saveConfig", component_property="style"),
              Input(component_id="interval_saveConfigSpan", component_property="n_intervals"),
              prevent_initial_call=True
              )
def hideSpanSaveConfig(n_intervals):
    return {'display': 'none'}


# Set timer to make save config span disappear
@app.callback(Output(component_id="interval_saveConfigSpan", component_property="n_intervals"),
              Input(component_id="button_saveConfig", component_property="n_clicks"),
              prevent_initial_call=True
              )
def setTimerSeaveConfig(n_clicks):
    return 0


if __name__ == '__main__':
    def sigint_handler(signal, frame):
        # Ignore sigint in main process
        print('CTRL-C pressed but ignored!')
    signal.signal(signal.SIGINT, sigint_handler)
    app.run(debug=True, host="0.0.0.0")
