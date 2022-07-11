import PySimpleGUI as sg 
from models import get_prediction
from helper import font, resize_image, resource_path

# Set theme
sg.theme(new_theme = 'Material2')

# Title 
title_text = [sg.Image(filename = resource_path('media/logo.png'), size = (64,64)), sg.Text('TB X-ray Classifier', font = font(27), auto_size_text = True)]

# file information
info_rows = [
    [
        sg.Text('Select Image:', font = font(), auto_size_text = True),
        sg.In(enable_events = True, key = 'file_input', font = font()),
        sg.FileBrowse(font = font())
    ],
    [
        sg.Text('File selected:', font = font(), auto_size_text = True), 
        sg.Text('', key = 'file_name', enable_events = True, font = font(), auto_size_text = True)
    ]
]

# Classification related information
classification_column = [
    [sg.Text('Classification:', font = font(), auto_size_text = True)],
    [sg.Text('', key = 'classification', enable_events = True, font = font(), auto_size_text = True)]
]

# Showcase image
image_column = [
    [sg.Text('Image chosen', font = font(), auto_size_text = True)],
    [sg.Image(filename = resource_path('media/empty.png'), key = 'image', size = (256,256))],
]

# Setup layout
layout = [
    title_text,
    info_rows,
    [
        sg.Column(image_column),
        sg.VSeperator(), 
        sg.Column(classification_column),
    ]
]

# Create window
window = sg.Window(title = 'TB X-ray Classifier', 
                    layout = layout,
                    size = (1000, 800),
                    element_padding = (20,20,20,20), 
                    resizable = True,
                    finalize = True)
window.set_min_size((1000,600))

# Start main loop
while True:
    event, values = window.read()
    
    if event == sg.WIN_CLOSED:
        break
    
    if event == 'file_input':
        # Read file path
        file_path = values['file_input']
        try:
            # Update filename in front end
            window['file_name'].update(file_path.split('/')[-1])
            
            # Load image and get prediction
            prediction = 'Potential Tuberculosis infection observed.' if get_prediction(file_path) else 'No Tuberculosis infection observed.'

            # Update classification information 
            window['classification'].update(prediction)
            
            # Add image 
            window['image'].update(data = resize_image(file_path),  size = (256,256))
        except Exception as e:
            # Make sound and show error
            window.ding(0)
            sg.popup_ok('Error!', e)

window.close()