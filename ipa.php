<?php
    
    $number = $_GET['number']; // номер телефона
    $text = $_GET['text']; // текст
    $url = "https://api.mobizon.kz/service/message/sendsmsmessage?recipient=".$number."&text=".$text."&apiKey=kzdb7f44270fae1f72d59037dfb50fa0e6d6f8484cc4893d81ab041387fe397b70b8fa";
    
    echo file_get_contents($url);
    
?>