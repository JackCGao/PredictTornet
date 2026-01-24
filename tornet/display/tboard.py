import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tornet.display.display import plot_radar


def log_image(data, score, filename, vars_to_plot, file_writer, step,):
    # Prepare the plot
    figure = create_image(data, score, filename, vars_to_plot)
    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image(filename, plot_to_image(figure), step=step)

        
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png',dpi=100)
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def create_image(data, score, filename, vars_to_plot):
    """
    Creates radar visualization
    """
    
    fig = plt.figure(figsize=(12,6))

    plot_radar(data,
                fig=fig,
                channels=vars_to_plot,
                include_cbar=True,
                time_idx=-1, # show last frame
                n_rows=2, n_cols=3)
    fname=os.path.basename(filename)
    fig.text(.5, .05,  '%s, score=%f' % (fname,score), ha='center')


    return fig