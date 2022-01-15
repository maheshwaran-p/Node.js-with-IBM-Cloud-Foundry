const express = require('express');
const multer = require('multer');
const uuid = require('uuid').v4;

const app = express();

const storage = multer.diskStorage({
    distination: (req, file, cb) => {
        cb(null, 'uploads');

    },
    filename: (req, file, cb) => {
        const { originalname } = file;
        cb(null, originalname);
    }
})

const upload = multer(storage);

app.use(express.static('Html'));

app.post('/upload', upload.single('MarkSheet'), (req, res) => {
    return res.json({ Status: 'OK' })
})

app.listen(2000, () => console.log('App is Listening on port 2000'));