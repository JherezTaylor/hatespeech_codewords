// Transporter transformer file
function transform(msg) {
    msg["ns"] = "doctext";
    return msg
}