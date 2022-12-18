import numpy as np
import ensembles
import pandas as pd
from flask import send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for
from flask import render_template, redirect

from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired, NumberRange, Optional
from wtforms import StringField, SubmitField, SelectField, IntegerField, FloatField

app = Flask(__name__, template_folder='html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
app.config["UPLOAD_FOLDER"] = app.root_path
Bootstrap(app)


class NameForm(FlaskForm):
    name = SelectField('Выберите модель', choices=['RandomForest', 'GradientBoosting'], validators=[DataRequired()])
    submit = SubmitField('Выбрать')


class DateForest(FlaskForm):
    n_estimators = IntegerField('Введите число деревьев', validators=[DataRequired()])
    max_depth = IntegerField('Введите максимальную глубину дерева(необязательно)',
                             validators=[Optional()])
    feature = FloatField('Введите долю (от 0 до 1)  признаков для каждого дерева (необязательно)',
                         validators=[Optional()])
    data_train = FileField('Загрузите файл для обучения в формате csv',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    data_val = FileField('Загрузите файл для валидации в формате csv(необязательно)',
                         validators=[FileAllowed(['csv'], 'Неверный формат'), Optional()])
    target = StringField('Название столбца с целевой переменной', validators=[DataRequired()])

    submit = SubmitField('Обучить модель')


class DateBoosting(FlaskForm):
    n_estimators = IntegerField('Введите число деревьев', validators=[DataRequired()])
    max_depth = IntegerField('Введите максимальную глубину дерева(необязательно)', validators=[Optional()])
    feature = FloatField('Введите долю (от 0 до 1)  признаков для каждого дерева (необязательно)',
                         validators=[(NumberRange(0, 1, 'Число вне диапазона')), Optional()])
    learn = FloatField('Введите темп обучения - learning_rate (необязательно)',
                       validators=[Optional(), (NumberRange(0, 1, 'Число вне диапазона'))])
    data_train = FileField('Загрузите файл для обучения в формате csv',
                           validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    data_val = FileField('Загрузите файл для валидации в формате csv(необязательно)',
                         validators=[FileAllowed(['csv'], 'Неверный формат'), Optional()])
    target = StringField('Название столбца с целевой переменной', validators=[DataRequired()])

    submit = SubmitField('Обучить модель')


class Test(FlaskForm):
    submit = SubmitField('Выполнить предсказания')


class TestDate(FlaskForm):
    data_val = FileField('Загрузите файл для предсказания в формате csv',
                         validators=[FileRequired(), FileAllowed(['csv'], 'Неверный формат')])
    submit = SubmitField('Выполнить предсказания')


class Bac(FlaskForm):
    submit = SubmitField('Начать сначала')


mod = ensembles.RandomForestMSE(1)


@app.route('/sentiment', methods=['GET', 'POST'])
def model():
    try:
        name_form = NameForm()

        if name_form.validate_on_submit():
            app.logger.info('On text: {0}'.format(name_form.name.data))
            if name_form.name.data == 'RandomForest':
                return redirect(url_for('forest'))
            else:
                return redirect(url_for('boosting'))
        return render_template('name_form.html', form=name_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('model'))


err = []
score_train = 0
score_val = 0


@app.route('/forest', methods=['GET', 'POST'])
def forest():
    try:
        global mod
        global score_train
        global score_val
        d = DateForest()
        if d.validate_on_submit():
            err.clear()
            app.logger.info('On t: {0}'.format(d.data_train.data))
            mod = ensembles.RandomForestMSE(d.n_estimators.data, d.max_depth.data, d.feature.data)
            f = d.data_train.data
            train = pd.read_csv(f)
            X = np.array(train.drop([d.target.data], axis=1))
            y = np.array(train[d.target.data])
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if d.data_val.data is not None:
                test_d = pd.read_csv(d.data_val.data)
                y_val = np.array(test_d[d.target.data])
                X_val = np.array(test_d.drop([d.target.data], axis=1))
            else:
                X_val = None
                y_val = None
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if X_val is not None:
                score_val, time_val, score_train = mod.fit(X, y, X_val, y_val, True)
            else:
                score_val = None
                score_train = mod.fit(X, y, X_val, y_val, True)
            app.logger.info('On t: {0}'.format(d))
            return redirect(url_for('choice', n_estimators=d.n_estimators.data, max_depth=d.max_depth.data,
                                    feature=d.feature.data))

        return render_template('form_forest.html', form=d, err=err)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        err.clear()
        err.append(exc)
        return redirect(url_for('forest'))


@app.route('/boosting', methods=['GET', 'POST'])
def boosting():
    try:
        d = DateBoosting()
        global mod
        global score_train
        global score_val
        if d.validate_on_submit():
            app.logger.info('On t: {0}'.format(d.data_train.data))
            if d.learn.data is None:
                mod = ensembles.GradientBoostingMSE(d.n_estimators.data,
                                                    max_depth=d.max_depth.data, feature_subsample_size=d.feature.data)
            else:
                mod = ensembles.GradientBoostingMSE(d.n_estimators.data, learning_rate=d.learn.data,
                                                    max_depth=d.max_depth.data, feature_subsample_size=d.feature.data)
            f = d.data_train.data
            train = pd.read_csv(f)
            X = np.array(train.drop([d.target.data], axis=1))
            y = np.array(train[d.target.data])
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if d.data_val.data is not None:
                test_d = pd.read_csv(d.data_val.data)
                y_val = np.array(test_d[d.target.data])
                X_val = np.array(test_d.drop([d.target.data], axis=1))
            else:
                X_val = None
                y_val = None
            app.logger.info('On t: {0}'.format(d.data_val.data))
            if X_val is not None:
                score_val, time_val, score_train = mod.fit(X, y, X_val, y_val, True)
            else:
                score_val = None
                score_train = mod.fit(X, y, X_val, y_val, True)
            app.logger.info('On t: {0}'.format(d))
            return redirect(url_for('choice_2', n_estimators=d.n_estimators.data, max_depth=d.max_depth.data,
                                    feature=d.feature.data, learn=d.learn.data))

        return render_template('form_boosting.html', form=d, err=err)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        err.clear()
        err.append(exc)
        return redirect(url_for('boosting'))


@app.route('/choice', methods=['GET', 'POST'])
def choice():
    try:
        app.logger.info('On t: {0}'.format(score_train))
        f_train = pd.DataFrame(score_train)
        app.logger.info('On t: {0}'.format(score_train))
        f_train.to_csv('train_pred.csv', index=False)
        name_train = 'train_pred.csv'
        app.logger.info('On val_la: {0}'.format(score_val))
        if score_val is not None:
            app.logger.info('On val: {0}'.format(score_val))
            f_val = pd.DataFrame(score_val)
            f_val.to_csv('val_pred.csv', index=False)
            name_val = 'val_pred.csv'
        else:
            name_val = None
        n_estimators = request.args.get('n_estimators')
        max_depth = request.args.get('max_depth')
        feature = request.args.get('feature')
        test_f = Test()
        if test_f.validate_on_submit():
            app.logger.info('On t: {0}'.format(mod))
            return redirect(url_for('test'))
        return render_template('result.html', n_estimators=n_estimators, max_depth=max_depth,
                               feature=feature, score_train=name_train, score_val=name_val, form=test_f)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('choice'))


@app.route('/choicebust', methods=['GET', 'POST'])
def choice_2():
    try:
        f_train = pd.DataFrame(score_train)
        f_train.to_csv('train_pred.csv', index=False)
        name_train = 'train_pred.csv'
        if score_val is not None:
            f_val = pd.DataFrame(score_val)
            f_val.to_csv('val_pred.csv', index=False)
            name_val = 'val_pred.csv'
        else:
            name_val = ''
        n_estimators = request.args.get('n_estimators')
        max_depth = request.args.get('max_depth')
        feature = request.args.get('feature')
        learn = request.args.get('learn')
        test_f = Test()
        if test_f.validate_on_submit():
            return redirect(url_for('test'))
        return render_template('result_boost.html', n_estimators=n_estimators, max_depth=max_depth, learn=learn,
                               feature=feature, score_train=name_train, score_val=name_val, form=test_f)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('choice_2'))


err_test = []


@app.route('/test', methods=['GET', 'POST'])
def test():
    try:
        form = TestDate()
        name = request.args.get('name')
        if form.validate_on_submit():
            f = form.data_val.data
            X = pd.read_csv(f)
            app.logger.info('Exc: {0}'.format(1))
            y_pred = mod.predict(np.array(X))
            file = pd.DataFrame(y_pred)
            app.logger.info('On t: {0}'.format(y_pred))
            file.to_csv('pred.csv', index=False)
            app.logger.info('Exc: {0}'.format(1))
            err_test.clear()
            return redirect(url_for('test', name='pred.csv'))
        bac = Bac()
        if bac.validate_on_submit():
            return redirect(url_for('model'))
        return render_template('test.html', form=form, name=name, form_bac=bac, err=err_test)
    except Exception as exc:
        err_test.clear()
        err_test.append(exc)
        app.logger.info('Exception: {0}'.format(err))
        return redirect(url_for('test'))


@app.route('/download/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('model'))
