class Order:
    # class attribute
    orders = []

    # instance attributes
    orderid = 0
    shipping_address = ''
    expedited = False
    shipped = False
    customer = None
 
    @static
    def get_expedited_orders_customer_names():
        output = []
        for order in Order.orders:
            if order.expedited:
                output.append(order.customer.name)
        return output
    @static
    def get_expedited_orders_customer_addresses(self):
        output = []
        for order in Order.orders:
            if order.expedited:
                output.append(order.customer.address)
        return output
    @static
    def get_expedited_orders_shipping_addresses(self):
        output = []
        for order in Order.orders:
            if order.expedited:
                output.append(order.shipping_address)
        return output
